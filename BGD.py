import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# Read the file CSV
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
ite = 2500
# Learning rate
learning_rate = 0.001
J_history = np.zeros(ite)
J_ite =  np.zeros(ite)

for i in range(0,ite):
    y_res = np.dot(X_train,theta) 
    error = y_res - y_train
    gradients = np.dot(X_train.T,error)/len(X_train)
    theta = theta - learning_rate * gradients
    J_history[i] = np.mean(np.square(X_train.dot(theta)-y_train))
    J_ite[i] = i

predictions = X_test.dot(theta)
y_test = y_test.reshape(-1,1)
mse = np.mean(np.square(predictions - y_test))
print("Mean Squared Error on Test Set:", mse)

plt.figure(figsize=(8, 6))
plt.plot(J_ite, J_history, marker='o', color='b', linestyle='-', linewidth=2, markersize=8, label='J')
plt.xlabel('Iterations')
plt.ylabel('Costs')
plt.title('Batch Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
