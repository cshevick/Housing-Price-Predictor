# Housing Price Prediction

This project predicts housing prices using both Linear Regression and Random Forest Regression models. The data is preprocessed, and hyperparameter tuning is performed using GridSearchCV to optimize the Random Forest model.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict the median house values based on various features such as location, number of rooms, population, etc. The models used for prediction are Linear Regression and Random Forest Regression. The Random Forest model is further optimized using hyperparameter tuning.

## Features

- Linear Regression model for baseline comparison.
- Random Forest Regression model for better performance.
- Hyperparameter tuning using GridSearchCV to optimize the Random Forest model.
- Evaluation of models using R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Dataset

The dataset used in this project is the California Housing dataset. It includes features such as `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and `ocean_proximity`.

## Preprocessing

Preprocessing steps include:
- Handling missing values.
- Encoding categorical variables.
- Applying logarithmic transformation to skewed features.
- Creating new features by combining existing ones.

## Model Training and Evaluation

The following models are trained and evaluated:
1. **Linear Regression:**
   - Trained on the preprocessed and scaled data.
   - Evaluated on the test set.
2. **Random Forest Regression:**
   - Trained on the preprocessed and scaled data.
   - Evaluated on the test set.
3. **Optimized Random Forest Regression:**
   - Hyperparameter tuning using GridSearchCV.
   - Trained and evaluated using the best parameters from the grid search.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction
