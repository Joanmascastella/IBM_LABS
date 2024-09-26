import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import requests

# Download the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
filename = "FuelConsumption.csv"

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File downloaded as {filename}")
    else:
        print(f"Failed to download file: {response.status_code}")

download_file(url, filename)

# Load the dataset
df = pd.read_csv(filename)

# Select relevant features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split the dataset into training and testing sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Prepare training and testing data
train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']]).ravel()  # ravel() to flatten array

test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']]).ravel()

# Standardize the features (mean = 0, variance = 1)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Initialize and fit the SGDRegressor model
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, learning_rate='adaptive', eta0=0.01)
sgd_reg.fit(train_x, train_y)

# Coefficients and intercept
print("Coefficients: ", sgd_reg.coef_)
print("Intercept: ", sgd_reg.intercept_)

# Predict on the test set
y_hat_sgd = sgd_reg.predict(test_x)

# Evaluation metrics
print("Mean absolute error: %.2f" % np.mean(np.abs(y_hat_sgd - test_y)))
print("Mean squared error: %.2f" % mean_squared_error(test_y, y_hat_sgd))
print("R2-score: %.2f" % r2_score(test_y, y_hat_sgd))


plt.scatter(test_y, y_hat_sgd, color='blue', label='Predicted vs Actual')
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")

# Adding a diagonal line for reference (perfect prediction line)
plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], color='red', linestyle='--', label='Perfect Prediction Line')

# Adding legend to the plot
plt.legend()

# Show the plot
plt.show()