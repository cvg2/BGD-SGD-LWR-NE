import pandas as pd
import random
import numpy as np

def weights_calculate(x0, X, tau):
    aux1 = np.sum(np.square(X - x0),axis=1)
    weight = np.exp(-aux1 / (2 * np.square(tau)) )
    return weight

def entrenar_LWR(x_consulta,X_train, y_train, tau):
    weight = weights_calculate(x_consulta, X_train, tau)
    aux1 = X_train.dot(theta) - y_train
    aux2 = weight.dot(aux1)
    aux3 = x_consulta.T.dot(aux2)
    gradient = aux3 / np.sum(weight)
    return gradient
    #gradient = (xi.T @ (weights * (X_b @ theta - y))) / np.sum(weights)

# Read the file CSV

data = pd.read_csv('CrabAgePrediction.csv')

# Get Data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

len_X = len(X)

min_value = X.min(axis=0, keepdims=True)
max_value = X.max(axis=0, keepdims=True)
X_scaled = (X - min_value) / (max_value - min_value)

# Add bias term to the input features
X_b = np.c_[np.ones((X.shape[0],1)),X_scaled]

num_training_instances = int(0.9 * len_X)
X_train = X_b[:num_training_instances]
y_train = y[:num_training_instances]
X_test = X_b[num_training_instances:]
y_test = y[num_training_instances:]
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

iterations = 1000
range_ite = int(iterations/2)

theta = np.zeros((9, 1))

# Learning rate
learning_rate = 0.001
ite=500
tau = 0.05
for j in range(ite):
    first_iteration = random.randint(range_ite, len(X_train) - range_ite + 1) - range_ite
    for i in range(first_iteration,first_iteration + iterations):
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        gradient = entrenar_LWR(xi,X_train,y_train, tau)
        gradient = gradient.reshape(-1,1)
        theta = theta - learning_rate * gradient

predictions = X_test.dot(theta)
mse = np.mean(np.square(predictions - y_test))

print("Mean Squared Error on Test: ", mse)
