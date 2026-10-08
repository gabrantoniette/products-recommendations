# E-commerce Recommendation System

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/products-recommendations-dark.svg">
  <img src="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/products-recommendations-light.svg" alt="Languages in products-recommendations, by share of code">
</picture>

A web application that recommends products to each user with a neural network trained in the browser using TensorFlow.js, based on their purchase history and age.

> This is a personal study project and is not production-ready.

## Project Structure

- `index.html` - Main HTML file for the application
- `src/index.js` - Entry point for the application
- `src/view/` - Contains classes for managing the DOM and templates
- `src/controller/` - Contains controllers to connect views and services
- `src/service/` - Contains business logic for data handling
- `src/workers/` - Web Worker that trains the model off the main thread
- `data/` - Contains JSON files with user and product data (fictional sample data)

## Setup and Run

1. Install dependencies:
```
npm install
```

2. Start the application:
```
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

4. Click **Train Model**, select a user and click **Run Recommendation**.

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage
- Model training in a Web Worker, with live loss and accuracy charts in tfjs-vis
- Products ranked for the selected user by the trained model

## How the Recommendations Work

1. Each product becomes a weighted feature vector: normalized price, normalized average age of its buyers, and one-hot category and color. Category weighs 0.4, color 0.3, price 0.2 and age 0.1.
2. Each user becomes the average of the vectors of the products they bought. A user with no purchases is represented by their normalized age only.
3. Every user and product pair is a training example, labeled 1 if the user bought the product and 0 if not.
4. A dense neural network (128 → 64 → 32 → 1, sigmoid output) learns to predict that label with binary cross-entropy.
5. To recommend, the model scores every product for the selected user and the list is sorted from highest to lowest score.

