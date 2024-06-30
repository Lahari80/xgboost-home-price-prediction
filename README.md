# xgboost-home-price-prediction
To predict home prices using XGBoost with factors such as income, schools, hospitals, and crime rates, you'll need to follow these steps:

Data Collection: Gather data on home prices and the factors affecting them. This may include:

Median household income
Quality and number of schools in the area
Proximity to hospitals
Crime rates
Data Preprocessing: Clean and preprocess the data, including handling missing values, encoding categorical variables, and scaling numerical features.

Feature Engineering: Create new features if necessary, such as average distance to the nearest school or hospital.

Model Training: Use the XGBoost library to train a model on the preprocessed data.

Model Evaluation: Evaluate the model's performance using appropriate metrics like RMSE (Root Mean Squared Error).

Model Tuning: Tune the model's hyperparameters to improve its performance.

Let's go through these steps with some example code.

Step 1: Data Collection
Assume you have a dataset home_prices.csv with columns: price, income, schools, hospitals, crime_rate.

Step 2: Data Preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('home_prices.csv')

# Separate features and target variable
X = data[['income', 'schools', 'hospitals', 'crime_rate']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Model Training
python
Copy code
# Initialize XGBoost model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
xgb.fit(X_train, y_train)
Step 4: Model Evaluation
python
Copy code
# Predict on the test set
y_pred = xgb.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
Step 5: Model Tuning
You can use GridSearchCV or RandomizedSearchCV from scikit-learn to tune the hyperparameters of the XGBoost model.

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize GridSearchCV
grid_search = GridSearchCV
