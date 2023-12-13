#Student Name: Joshua Salas
#Date: October 14, 2023

import pandas as pd
from sklearn.model_selection import train_test_split
import math
from collections import Counter

# 1. Load dataset
def load_dataset(filename):
    df = pd.read_csv(filename)
    return df

data = load_dataset('lab2_dataset/q2/Iris.csv')
X = data.drop(columns=['Species']).values
y = data['Species'].values

# 2. Split dataset into training and test sets using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# 3. Define the kNN algorithm
def euclidean_distance(instance1, instance2):
    """Calculate the Euclidean distance between two instances."""
    #Using math library to give euclidean distance
    euclid_dist = math.dist(instance1, instance2)
    return euclid_dist


def get_neighbors(X_train, test_instance, k):
    """Get the k nearest neighbors for a test instance."""
    distances = []
    neighbors = []
    
    #Append the euclidean distance and the class onto distances
    for row in range(len(X_train)):
        distances.append([euclidean_distance(X_train[row,:],test_instance), y[row]])

    #sort distances by euclidean distance
    distances = sorted(distances, key=lambda x:(x[0]))

    #return only the neighbor's class now that it is sorted
    neighbors = [distances[i][1] for i in range(len(distances))]
    
    return neighbors[:k]

def get_response(neighbors, y_train):
    """Determine the class label for a current instance based on the majority 
    class label of its k neighbors."""
    prediction = None

    #return the largest counted class found in the k-nearest neighbors
    prediction = max(Counter(neighbors))    

    return prediction

# 4. Use the kNN algorithm to predict the class labels of the test set
k = 3
predictions = []
for current_instance in X_test:
    neighbors = get_neighbors(X_train, current_instance, k)
    prediction = get_response(neighbors, y_train)
    predictions.append(prediction)

# 5. Calculate the accuracy of the predictions
correct = sum([y_true == y_pred for y_true, y_pred in zip(y_test, predictions)])
accuracy = (correct / len(y_test)) * 100.0
print(f"Accuracy: {accuracy:.2f}%")