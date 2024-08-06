# Housing-Price-Predictor
Machine Learning Housing Price Predictor (Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn)  

# Housing Price Prediction

This project predicts housing prices using a Random Forest Regression model. The data is preprocessed, and hyperparameter tuning is performed using GridSearchCV to optimize the model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict the median house values based on various features such as location, number of rooms, population, etc. The model is built using Random Forest Regression, and various preprocessing steps are applied to clean and transform the data.

## Dataset

The dataset used in this project is the California Housing dataset. It includes features such as `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and `ocean_proximity`.

## Preprocessing

Preprocessing steps include:
- Handling missing values.
- Encoding categorical variables.
- Applying logarithmic transformation to skewed features.
- Creating new features by combining existing ones.

## Model

A Random Forest Regression model is used for prediction. The model is trained and evaluated using the preprocessed data.

## Hyperparameter Tuning

GridSearchCV is used to perform hyperparameter tuning. The following parameters are optimized:
- `n_estimators`
- `max_features`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `bootstrap`

## Evaluation

The model is evaluated using the following metrics:
- R² score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Requirements

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the required libraries using:
```bash
pip install pandas numpy matplotlib scikit-learn
