# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and Prepare Data
# Replace 'data.csv' with your actual data file
data = pd.read_csv("data.csv")

# Display the first few rows to check the data structure
print(data.head())

# Extract features and target
# Assuming data columns: 'preparation_time', 'distance', 'traffic', 'weather', 'driver_availability', 'delivery_time'
X = data[['preparation_time', 'distance', 'traffic', 'weather', 'driver_availability']]
y = data['delivery_time']

# Encode Categorical Variables (for traffic and weather)
X = pd.get_dummies(X, columns=['traffic', 'weather'], drop_first=True)

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Calculate Error Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display Results
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)
