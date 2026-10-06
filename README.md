# E-commerce Recommendation System

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/products-recommendations-dark.svg">
  <img src="https://raw.githubusercontent.com/gabrantoniette/gabrantoniette/main/assets/generated/languages/products-recommendations-light.svg" alt="Languages in products-recommendations, by share of code">
</picture>

A web application that displays user profiles and product listings, with the ability to track user purchases for future machine learning recommendations using TensorFlow.js.

> **Work in progress.** This is a personal study project and is not production-ready.

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

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage

## Future Enhancements

- TensorFlow.js-based recommendation engine
- User similarity analysis
- Product recommendation based on purchase history

## Credits and License

This project is based on the
[`exemplo-01-ecommerce-recomendations-template`](https://github.com/unipds-engenharia-de-ia-aplicada/engenharia-de-software-com-ia-aplicada/tree/main/modulo01-fundamentos-de-ia-e-llms-para-programadores/exemplo-01-ecommerce-recomendations-template)
from the [Engenharia de Software com IA Aplicada](https://github.com/unipds-engenharia-de-ia-aplicada/engenharia-de-software-com-ia-aplicada)
course by [Unipds Educação](https://unipds.com.br/org-pos-ia/), used for study purposes.

Changes from the original template:

- Fixed the `npm start` script so it runs on Windows and watches `src/`.
- Added Subresource Integrity hashes to the CDN assets in `index.html`.

The original material is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE.md](LICENSE.md).
This repository is not affiliated with or endorsed by Unipds Educação.
