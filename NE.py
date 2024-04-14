import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# Read file CSV
data = pd.read_csv('CrabAgePrediction.csv')

# Get data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
len_X = len(X)
	
# Add bias term to the input features
X_b = np.c_[np.ones((X.shape[0],1)),X]

num_training_instances = int(0.9 * len_X)
X_train = X_b[:num_training_instances]
y_train = y[:num_training_instances]
X_test = X_b[num_training_instances:]
y_test = y[num_training_instances:]


y_train = y_train.reshape(-1, 1) 
theta = np.zeros((9, 1))

theta1 = np.dot(X_train.T, X_train)
theta1 = np.linalg.inv(theta1)
theta2 = np.dot(X_train.T, y_train)
theta = np.dot(theta1, theta2)

predictions = X_test.dot(theta)
y_test = y_test.reshape(-1,1)
mse = np.mean(np.square(predictions - y_test))
print("Mean Squared Error on Test Set:", mse)

