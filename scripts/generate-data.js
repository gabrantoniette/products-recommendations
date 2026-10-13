// Generates data/products.json and data/users.json: 25 products and 50,000
// fictional customers whose purchases follow known tendencies, so the model
// has a pattern to learn. Pure random purchases would teach it nothing.
// The same seed always writes the same files.
//
// Usage: node scripts/generate-data.js [--users 50000] [--seed 42]
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const PRODUCTS = [
    { id: 1, name: 'Fones de Ouvido Sem Fio', category: 'eletrônicos', price: 129.99, color: 'preto' },
    { id: 2, name: 'Relógio Inteligente', category: 'eletrônicos', price: 199.99, color: 'prata' },
    { id: 3, name: 'Caixa de Som Bluetooth', category: 'eletrônicos', price: 89.99, color: 'azul' },
    { id: 4, name: 'Camiseta Estampada', category: 'vestuário', price: 49.99, color: 'branco' },
    { id: 5, name: 'Calça Jeans Slim', category: 'vestuário', price: 99.99, color: 'azul' },
    { id: 6, name: 'Tênis Esportivo', category: 'calçados', price: 149.99, color: 'vermelho' },
    { id: 7, name: 'Sandália Casual', category: 'calçados', price: 69.99, color: 'bege' },
    { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
    { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' },
    { id: 10, name: 'Óculos de Sol', category: 'acessórios', price: 89.99, color: 'marrom' },
    { id: 11, name: 'Teclado Mecânico', category: 'eletrônicos', price: 249.99, color: 'preto' },
    { id: 12, name: 'Mouse Sem Fio', category: 'eletrônicos', price: 79.99, color: 'cinza' },
    { id: 13, name: 'Carregador Portátil', category: 'eletrônicos', price: 99.99, color: 'branco' },
    { id: 14, name: 'Webcam Full HD', category: 'eletrônicos', price: 189.99, color: 'preto' },
    { id: 15, name: 'Jaqueta Corta-Vento', category: 'vestuário', price: 189.99, color: 'verde' },
    { id: 16, name: 'Moletom com Capuz', category: 'vestuário', price: 139.99, color: 'cinza' },
    { id: 17, name: 'Camisa Social', category: 'vestuário', price: 119.99, color: 'azul' },
    { id: 18, name: 'Shorts de Corrida', category: 'vestuário', price: 59.99, color: 'preto' },
    { id: 19, name: 'Bota de Couro', category: 'calçados', price: 279.99, color: 'marrom' },
    { id: 20, name: 'Chinelo', category: 'calçados', price: 29.99, color: 'azul' },
    { id: 21, name: 'Sapatênis', category: 'calçados', price: 169.99, color: 'bege' },
    { id: 22, name: 'Tênis de Corrida', category: 'calçados', price: 229.99, color: 'branco' },
    { id: 23, name: 'Carteira de Couro', category: 'acessórios', price: 89.99, color: 'marrom' },
    { id: 24, name: 'Cinto de Couro', category: 'acessórios', price: 59.99, color: 'preto' },
    { id: 25, name: 'Relógio Analógico', category: 'acessórios', price: 319.99, color: 'prata' },
];

const FIRST_NAMES = [
    'Ana', 'Bruno', 'Camila', 'Diego', 'Eduarda', 'Felipe', 'Gabriela', 'Henrique', 'Isabela', 'João',
    'Larissa', 'Lucas', 'Mariana', 'Mateus', 'Natália', 'Otávio', 'Patrícia', 'Rafael', 'Renata', 'Rodrigo',
    'Sabrina', 'Thiago', 'Vanessa', 'Vinícius', 'Beatriz', 'Carlos', 'Daniela', 'Fernando', 'Helena', 'Igor',
    'Juliana', 'Leonardo', 'Letícia', 'Marcelo', 'Paula', 'Pedro', 'Roberta', 'Sérgio', 'Tatiana', 'Yasmin',
];

const LAST_NAMES = [
    'Almeida', 'Barbosa', 'Cardoso', 'Costa', 'Dias', 'Ferreira', 'Gomes', 'Lima', 'Martins', 'Melo',
    'Moreira', 'Nunes', 'Oliveira', 'Pereira', 'Ribeiro', 'Rocha', 'Santos', 'Silva', 'Souza', 'Teixeira',
];

export const MIN_AGE = 18;
export const MAX_AGE = 70;
export const SHARE_WITHOUT_PURCHASES = 0.2;
export const MAX_PURCHASES = 8;

// How much each age group leans toward each category: a multiplier on the
// chance of buying a product of that category. Price fit and favorite colors
// multiply in too, so it is a lean, not a rule.
export const AGE_GROUPS = [
    { maxAge: 29, categories: { 'eletrônicos': 3, 'calçados': 2.5, 'vestuário': 1, 'acessórios': 0.6 } },
    { maxAge: 49, categories: { 'vestuário': 3, 'eletrônicos': 2, 'acessórios': 1.2, 'calçados': 1 } },
    { maxAge: MAX_AGE, categories: { 'acessórios': 3, 'vestuário': 2.5, 'calçados': 1, 'eletrônicos': 0.6 } },
];

const COLORS = [...new Set(PRODUCTS.map(product => product.color))];
const FAVORITE_COLOR_BOOST = 3;
const PRICE_SPREAD = 60;

// Small, seedable random number generator (mulberry32), so the files are reproducible.
export function randomGenerator(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6D2B79F5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const pick = (random, list) => list[Math.floor(random() * list.length)];

// Box-Muller: a normally distributed number with mean 0 and deviation 1.
const gaussian = random => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

// The hidden traits behind a customer's purchases. They never go into
// users.json: the model only sees what the customer bought.
export function customerTraits(random, age) {
    // Older customers spend more, with plenty of variation between people.
    const budget = Math.min(350, Math.max(30, 60 + (age - MIN_AGE) * 3 + gaussian(random) * 40));
    const first = pick(random, COLORS);
    const rest = COLORS.filter(color => color !== first);
    return { budget, favoriteColors: [first, pick(random, rest)] };
}

// The weight a product gets when a customer with these traits picks a purchase.
export function purchaseWeight(product, age, traits) {
    const group = AGE_GROUPS.find(group => age <= group.maxAge);
    const category = group.categories[product.category];
    const priceFit = Math.exp(-0.5 * ((product.price - traits.budget) / PRICE_SPREAD) ** 2) + 0.1;
    const color = traits.favoriteColors.includes(product.color) ? FAVORITE_COLOR_BOOST : 1;
    return category * priceFit * color;
}

function pickPurchases(random, age, traits, count) {
    const available = [...PRODUCTS];
    const purchases = [];
    while (purchases.length < count) {
        const weights = available.map(product => purchaseWeight(product, age, traits));
        let draw = random() * weights.reduce((sum, weight) => sum + weight, 0);
        let index = 0;
        while (draw > weights[index] && index < weights.length - 1) {
            draw -= weights[index];
            index++;
        }
        purchases.push({ ...available[index] });
        available.splice(index, 1);
    }
    return purchases;
}

export function generate({ users = 50000, seed = 42 } = {}) {
    const random = randomGenerator(seed);
    const customers = [];
    const traits = [];

    for (let id = 1; id <= users; id++) {
        const age = MIN_AGE + Math.floor(random() * (MAX_AGE - MIN_AGE + 1));
        const hidden = customerTraits(random, age);
        const count = random() < SHARE_WITHOUT_PURCHASES
            ? 0
            : 1 + Math.floor(random() ** 1.5 * MAX_PURCHASES);

        customers.push({
            id,
            name: `${pick(random, FIRST_NAMES)} ${pick(random, LAST_NAMES)}`,
            age,
            purchases: pickPurchases(random, age, hidden, count),
        });
        traits.push(hidden);
    }

    return { products: PRODUCTS, users: customers, traits };
}

function readOption(name, fallback) {
    const index = process.argv.indexOf(`--${name}`);
    if (index === -1) return fallback;
    const value = Number(process.argv[index + 1]);
    if (!Number.isInteger(value) || value < 1) throw new Error(`--${name} needs a positive whole number`);
    return value;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const { products, users } = generate({
        users: readOption('users', 50000),
        seed: readOption('seed', 42),
    });
    const dataDir = new URL('../data/', import.meta.url);

    writeFileSync(new URL('products.json', dataDir), JSON.stringify(products, null, 4) + '\n');
    // One customer per line keeps 50,000 of them readable in a diff and small on disk.
    writeFileSync(
        new URL('users.json', dataDir),
        '[\n' + users.map(user => '    ' + JSON.stringify(user)).join(',\n') + '\n]\n'
    );

    const buyers = users.filter(user => user.purchases.length).length;
    const purchases = users.reduce((sum, user) => sum + user.purchases.length, 0);
    console.log(`${products.length} products, ${users.length} customers (${buyers} with purchases), ${purchases} purchases`);
}
