import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Read the CSV file
data = pd.read_csv('housing.csv')

# Ensure there are no missing values
data.dropna(inplace=True)

# Split the data into features and target
x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Preprocess the data function
def preprocess_data(df):
    df = df.join(pd.get_dummies(df.ocean_proximity).astype(int)).drop(['ocean_proximity'], axis=1)
    df['total_rooms'] = np.log(df['total_rooms'] + 1)
    df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
    df['population'] = np.log(df['population'] + 1)
    df['households'] = np.log(df['households'] + 1)
    df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    return df

# Preprocess the training, validation, and test data
train_data = preprocess_data(X_train.join(y_train))
val_data = preprocess_data(X_val.join(y_val))
test_data = preprocess_data(X_test.join(y_test))

# Ensure the validation and test data have the same columns as the training data
missing_cols = set(train_data.columns) - set(val_data.columns)
for col in missing_cols:
    val_data[col] = 0
val_data = val_data[train_data.columns]

missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns]

# Split preprocessed data into features and target
x_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
x_val, y_val = val_data.drop(['median_house_value'], axis=1), val_data['median_house_value']
x_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Train the LinearRegression model
linear_model = LinearRegression()
linear_model.fit(x_train_scaled, y_train)

# Evaluate LinearRegression on test set
test_predictions_lr = linear_model.predict(x_test_scaled)
test_score_lr = linear_model.score(x_test_scaled, y_test)
test_mae_lr = mean_absolute_error(y_test, test_predictions_lr)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, test_predictions_lr))

print("\nLinear Regression test score (R²):", test_score_lr)
print(f"Test MAE: {test_mae_lr}")
print(f"Test RMSE: {test_rmse_lr}")

# Train the RandomForestRegressor model
forest = RandomForestRegressor()
forest.fit(x_train_scaled, y_train)

# Evaluate RandomForestRegressor on test set
test_predictions_rf = forest.predict(x_test_scaled)
test_score_rf = forest.score(x_test_scaled, y_test)
test_mae_rf = mean_absolute_error(y_test, test_predictions_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test, test_predictions_rf))

print("\nRandom Forest Regression test score (R²):", test_score_rf)
print(f"Test MAE: {test_mae_rf}")
print(f"Test RMSE: {test_rmse_rf}")

# Perform hyperparameter tuning with GridSearchCV
param_grid = { "n_estimators": [375,425],
               "max_features": [40,45],
               "max_depth": [23,28],
               "min_samples_split": [3,4],
               "min_samples_leaf": [2],
               "bootstrap": [True]
            }

# Initialize and fit the GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
grid_search.fit(x_train_scaled, y_train)

# Get the best estimator
best_forest = grid_search.best_estimator_
best_forest.fit(x_train_scaled, y_train)

# Evaluate the best model on the test set
test_predictions_best_rf = best_forest.predict(x_test_scaled)
test_score_best_rf = best_forest.score(x_test_scaled, y_test)
test_mae_best_rf = mean_absolute_error(y_test, test_predictions_best_rf)
test_rmse_best_rf = np.sqrt(mean_squared_error(y_test, test_predictions_best_rf))

print("\nBest Random Forest Regression test score (R²):", test_score_best_rf)
print(f"Test MAE: {test_mae_best_rf}")
print(f"Test RMSE: {test_rmse_best_rf}")
print("Best Parameters:", grid_search.best_params_)
