import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
import requests

# Download the dataset from the given URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
filename = "teleCust1000t.csv"

# Function to download the dataset and save it locally
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"File downloaded as {filename}")
    else:
        print(f"Failed to download file: {response.status_code}")

# Download the file
download_file(url, filename)

# Load the dataset into a pandas DataFrame
df = pd.read_csv(filename)

# Define the feature matrix (X) and the target vector (y)
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values  # Customer category (target variable)

# Standardize the feature matrix X
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Initialize the K-Nearest Neighbors classifier with k=4
k = 57
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Make predictions on the test set
yhat = neigh.predict(X_test)

# Calculate and print the accuracy on the training and test sets
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# Scatter plot of actual vs predicted values (using y_test and yhat)
plt.figure(figsize=(8, 6))

# Scatter plot: X-axis -> Actual values (y_test), Y-axis -> Predicted values (yhat)
plt.scatter(y_test, yhat, c='blue', label='Predicted vs Actual', alpha=0.6)

# Plot a diagonal line where predicted = actual (ideal case)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')

# Set labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - KNN Classifier')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Tuning the value of k
Ks = 20 # Maximum number of neighbors to test
mean_acc = np.zeros((Ks-1))  # Array to store accuracy for each k
std_acc = np.zeros((Ks-1))  # Array to store standard deviation of accuracy
logloss_vals = np.zeros((Ks-1))  # Array to store log loss for each k

# Loop through values of k from 1 to Ks
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    
    # Get predicted probabilities for log loss
    yhat_prob = neigh.predict_proba(X_test)
    
    # Calculate log loss and accuracy
    logloss_vals[n-1] = log_loss(y_test, yhat_prob)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Plot accuracy for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(1, Ks), mean_acc, 'g', label='Accuracy')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.title('K-Nearest Neighbors Accuracy by Number of Neighbors')
plt.legend()
plt.show()

# Plot log loss for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(1, Ks), logloss_vals, 'r', label='Log Loss')
plt.ylabel('Log Loss')
plt.xlabel('Number of Neighbors (k)')
plt.title('K-Nearest Neighbors Log Loss by Number of Neighbors')
plt.legend()
plt.show()

# Print the best accuracy, log loss, and corresponding k values
best_k_acc = mean_acc.argmax() + 1
best_k_logloss = logloss_vals.argmin() + 1
print(f"The best accuracy is {mean_acc.max()} with k={best_k_acc}")
print(f"The lowest log loss is {logloss_vals.min()} with k={best_k_logloss}")