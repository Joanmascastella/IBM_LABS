import numpy as np
import pandas as pd
import sklearn.tree as tree
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydotplus
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Download the dataset from the given URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
filename = "decisiontree.csv"

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

# Load dataset into a variable so it can later be transformed and trained on 
df = pd.read_csv(filename)
print(df.shape)

# Pre-process the data, define the feature matrix and response vector 
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Encode categorical variables
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = df["Drug"]

# Split the data into training and test sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&','Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X test set {}'.format(X_testset.shape),'&','Size of y test set {}'.format(y_testset.shape))

# Train the decision tree model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Make predictions on the test set
predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# Export the decision tree to a .dot file and visualize it
dot_data = export_graphviz(drugTree, out_file=None, 
                           filled=True, rounded=True,
                           feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
                           class_names=drugTree.classes_,
                           special_characters=True)

# Generate a graph from dot data using pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)

# Create a PNG image from the graph and load it
png_image = graph.create_png()

# Display the PNG image using PIL and matplotlib
image = Image.open(BytesIO(png_image))

# Plot the image using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()