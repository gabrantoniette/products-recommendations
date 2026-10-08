import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');

let _globalCtx = {};
let _model = null;

const WEIGHTS = {
    age: 0.1,
    price: 0.2,
    category: 0.4,
    color: 0.3
}

const normalize = (value, min, max) => (value - min) / (max - min);

async function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const price = products.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

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


function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length) //only consider users with purchases
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name
                ) ? 1:0
                inputs.push([...userVector, ...productVector]) 
                //combine user vector and product vector into a single input vector
                labels.push(label)
            })
        })
    
    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2 //size of user vector + size of product vector
    }
}


async function configureNeuralNetandTrain(trainData) {

    const model = tf.sequential()

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )  

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    //above: input and the following added layers are incremental for a better learning process, 
    //allowing the model to capture more complex patterns in the data.

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
        epochs: 100,
        batchSize: 32,
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
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });

    const products = await (await fetch('/data/products.json')).json()

    const context = await makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })
    
    _globalCtx = context;
    
    const trainData = createTrainingData(context)
    _model = await configureNeuralNetandTrain(trainData)

    
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });


}

function recommend({ user }) {
    
    if(!_model) return;
    const context = _globalCtx;

    const userVector = encodeUser(user, _globalCtx).dataSync()
    const inputs = context.productVectors.map(({vector}) => {
        return [...userVector, ...vector]
    })

    const inputTensor = tf.tensor2d(inputs)
    const predictions = _model.predict(inputTensor)

    const scores = predictions.dataSync()
    
    const recommendations = context.productVectors.map((product, index) => {
        return {
            ...product.meta,
            name: product.name,
            score: scores[index] //prediction score for the product
        }
    })

    const sortedProducts = recommendations.sort((a, b) => b.score - a.score)

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
