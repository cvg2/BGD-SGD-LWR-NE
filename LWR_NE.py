import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def weights_calculate(x0, X, tau):
    aux1 = np.sum(np.square(X - x0),axis=1)
    weight = np.exp(-aux1 / (2 * np.square(tau)) )
    return weight

def entrenar_LWR(x_consulta,X_train, y_train, tau):
    weight = weights_calculate(x_consulta, X_train, tau)
    x_w = X_train.T * weight
    aux1 = np.linalg.pinv(x_w.dot(X_train))
    aux2 = aux1.dot(x_w)
    theta = aux2.dot(y_train)
    return theta

# Read file CSV
data = pd.read_csv('CrabAgePrediction.csv')

# Get data
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
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1, 1) 

theta = np.zeros((9, 1))
tau = 0.05
y_pred = []
for i in range(len(X_test)):
    x_consulta = X_test[i]
    theta = entrenar_LWR(x_consulta, X_train, y_train, tau)
    y_pred.append(np.dot(x_consulta,theta))

mse = np.mean(np.square(y_pred - y_test))
print("Mean Squared Error on Test Set",mse)
