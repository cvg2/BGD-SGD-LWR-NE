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
y_train = y_train.reshape(-1,1)

iterations = 1500
range_ite = int(iterations/2)

theta = np.zeros((9, 1))

# Learning rate
learning_rate = 0.00001

J_history = []
J_ite =  []
ite=1000
for j in range(ite):
    first_iteration = random.randint(range_ite, len_X - range_ite + 1) - range_ite
    for i in range(first_iteration,first_iteration + iterations):
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        yi = yi.reshape(-1,1)
        y_res = xi.dot(theta)
        y_res = y_res.reshape(-1,1)
        gradients = xi.T.dot(y_res-yi)
        theta = theta - learning_rate * gradients
    y_pred = X_train.dot(theta)
    J_history.append(np.mean(np.square(y_pred - y_train)))
    J_ite.append(j)

predictions = X_test.dot(theta)
y_test = y_test.reshape(-1,1)
mse = np.mean(np.square(predictions - y_test))

print("Mean Squared Error on Test Set:", mse)

plt.figure(figsize=(8, 6))
plt.plot(J_ite, J_history, marker='o', color='b', linestyle='-', linewidth=2, markersize=8, label='J')
plt.xlabel('Iterations')
plt.ylabel('Costs')
plt.title('Stochastic Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

