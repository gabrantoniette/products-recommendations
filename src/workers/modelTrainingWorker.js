import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');

let _globalCtx = {};
let _model = null;
let _training = false;

const WEIGHTS = {
    age: 0.1,
    price: 0.2,
    category: 0.4,
    color: 0.3
}

// About a million training pairs: a few passes with large batches is enough,
// where the 50 pairs of the first dataset needed 100 passes of 32.
const TRAINING = {
    epochs: 10,
    batchSize: 512
}

const normalize = (value, min, max) => (value - min) / (max - min);

async function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const price = products.map(p => p.price);

    // reduce instead of Math.min(...ages): spreading 50,000 ages as arguments can overflow the call stack
    const minAge = ages.reduce((min, age) => Math.min(min, age), Infinity);
    const maxAge = ages.reduce((max, age) => Math.max(max, age), -Infinity);

    const minPrice = Math.min(...price);
    const maxPrice = Math.max(...price);

    const colors = [... new Set(products.map(c => c.color))];
    const categories = [... new Set(products.map(c => c.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index];
        })
    );

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index];
        })
    );

    // age average, it helps to customize the recommendation based on the user's age.
    const ageAvg = (minAge + maxAge) / 2;
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    });
    
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
            ageSums[product.name] / ageCounts[product.name] :
            ageAvg

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )
    
    return {
        products,
        users,
        colorIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        productAvgAgeNorm,
        maxPrice,
        numCategories: categories.length,
        NumColors: colors.length,
        // price + age + categories + colors
        dimentions: 2 + categories.length + colors.length
    }
}


const oneHotWeighted = (index, length, weight) => 
    tf.oneHot(index, length).cast('float32').mul(weight)


function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price, 
            context.minPrice, 
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        context.productAvgAgeNorm[product.name] * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.NumColors,
        WEIGHTS.color
    )
    
    return tf.concat1d([price, age, category, color])

}


function encodeUser(user, context) {
    if(user.purchases.length) {
        return tf.stack(user.purchases.map(
            product => encodeProduct(product, context)
            )
        )
        .mean(0)
        .reshape([1, context.dimentions])
    }
    
    return tf.concat1d([
        tf.zeros([1]), //price ignored for users, as we don't have a specific product to reference
        tf.tensor1d([
            normalize(
                user.age,
                context.minAge,
                context.maxAge
            ) * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]), //category ignored for users
        tf.zeros([context.NumColors])      //color ignored for users
    ])
    .reshape([1, context.dimentions])
}


// The same average encodeUser takes for a user with purchases, built from the
// product vectors encoded once in trainModel instead of new tensors per purchase.
function averageOfPurchases(user, context) {
    const dims = context.dimentions
    const vectorByName = context.productVectorByName ??= Object.fromEntries(
        context.productVectors.map(({ name, vector }) => [name, vector])
    )
    const average = new Float32Array(dims)
    user.purchases.forEach(purchase => {
        const vector = vectorByName[purchase.name]
        for (let i = 0; i < dims; i++) average[i] += vector[i]
    })
    for (let i = 0; i < dims; i++) average[i] /= user.purchases.length
    return average
}


// The user vector without one product: the average of the other purchases, or
// the age-only vector a user with no purchases gets when it was the only one.
function tasteWithout(user, productName, context) {
    const others = { ...user, purchases: user.purchases.filter(purchase => purchase.name !== productName) }
    return others.purchases.length
        ? averageOfPurchases(others, context)
        : tf.tidy(() => encodeUser(others, context).dataSync())
}


function createTrainingData(context) {
    // 40,000 users with purchases times 25 products is about a million pairs, too
    // many to create tensors for each one. Every product is encoded once (in
    // trainModel), every user once, and the pairs are copied into one Float32Array.
    const dims = context.dimentions
    const inputDimention = dims * 2 //size of user vector + size of product vector
    const users = context.users.filter(u => u.purchases.length) //only consider users with purchases
    const rows = users.length * context.productVectors.length
    const xs = new Float32Array(rows * inputDimention)
    const ys = new Float32Array(rows)

    let row = 0
    users.forEach(user => {
        const userVector = averageOfPurchases(user, context)
        const bought = new Set(user.purchases.map(purchase => purchase.name))
        context.productVectors.forEach(({ name, vector }) => {
            // A bought product is learned from the user's OTHER purchases. With the full
            // average, the product itself is inside the user vector, and the model learns
            // "already in the history" instead of the taste that led to the purchase.
            const rowVector = bought.has(name) ? tasteWithout(user, name, context) : userVector
            //combine user vector and product vector into a single input vector
            xs.set(rowVector, row * inputDimention)
            xs.set(vector, row * inputDimention + dims)
            ys[row] = bought.has(name) ? 1 : 0
            row++
        })
    })

    return {
        xs: tf.tensor2d(xs, [rows, inputDimention]),
        ys: tf.tensor2d(ys, [rows, 1]),
        inputDimention
    }
}


async function configureNeuralNetandTrain(trainData) {

    const model = tf.sequential()

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 32,
            activation: 'relu'
        })
    )
    //one hidden layer of 32: on hidden purchases, the former 128 -> 64 -> 32 stack (14,337
    //weights against 1,025) ranked about 1 point better in the top 5 and trained slower.
    //See "How It Was Measured" in the README.

    model.add(
        tf.layers.dense({
            units: 1,
            activation: 'sigmoid' //compress the output to a range between 0 and 1, 
                                  //which is suitable for binary classification tasks.
                                  //0.9 is a strong recommendation, 0.1 is a weak recommendation.
        })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })
    
    await model.fit(trainData.xs, trainData.ys, {
        epochs: TRAINING.epochs,
        batchSize: TRAINING.batchSize,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;

}


async function trainModel({ users }) {
    // training now takes minutes: a second request while one runs is ignored
    if (_training) return;
    _training = true;
    console.log('Training model with users:', users.length)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });

    try {
        const products = await (await fetch('/data/products.json')).json()

        const context = await makeContext(products, users)
        context.productVectors = products.map(product => {
            return {
                name: product.name,
                meta: {...product},
                vector: tf.tidy(() => encodeProduct(product, context).dataSync())
            }
        })

        const known = new Set(products.map(product => product.name))
        const unknown = users.flatMap(user => user.purchases).find(purchase => !known.has(purchase.name))
        if (unknown) throw new Error(`purchase of a product not in products.json: ${unknown.name}`)

        const trainData = createTrainingData(context)
        const model = await configureNeuralNetandTrain(trainData)
        tf.dispose([trainData.xs, trainData.ys])

        //the previous model keeps answering recommendations until the new one is ready
        if (_model) {
            _model.optimizer.dispose()
            _model.dispose()
        }
        _model = model
        _globalCtx = context;

        postMessage({ type: workerEvents.trainingComplete });
    } catch (error) {
        console.error('Training failed:', error)
    } finally {
        _training = false;
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    }
}

function recommend({ user }) {

    if(!_model) return;
    const context = _globalCtx;

    const userVector = user.purchases.length
        ? averageOfPurchases(user, context)
        : tf.tidy(() => encodeUser(user, context).dataSync())
    const inputs = context.productVectors.map(({vector}) => {
        return [...userVector, ...vector]
    })

    const inputTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputTensor)

    const scores = predictions.dataSync()
    tf.dispose([inputTensor, predictions])
    
    const recommendations = context.productVectors.map((product, index) => {
        return {
            ...product.meta,
            name: product.name,
            score: scores[index] //prediction score for the product
        }
    })

    // what the user already bought goes to the end: the model was trained to score
    // products missing from the history, so their scores say nothing
    const owned = new Set(user.purchases.map(purchase => purchase.name))
    const sortedProducts = recommendations.sort(
        (a, b) => owned.has(a.name) - owned.has(b.name) || b.score - a.score
    )

    postMessage({ 
        type: workerEvents.recommend, 
        user, 
        recommendations: sortedProducts 
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
