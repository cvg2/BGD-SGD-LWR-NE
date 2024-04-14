import re
import numpy as np
import pandas as pd

normalfiles = np.array([2,3,4,5,6,7,8,9,10,12,14,15,16,17,19,20,21,22,23,24])
spamfiles = np.array([1,5,6,7,9,13,14,17,18,19,20,22])

extension = ".txt"
palabras_unicas = []

fpath = "normal/"
for numero in normalfiles:
    nombre_archivo = fpath + str(numero) + extension
    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip()
                if palabra not in palabras_unicas:
                    palabras_unicas.append(palabra)

fpath = "spam/"
for numero in spamfiles:
    nombre_archivo = fpath + str(numero) + extension
    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip()
                if palabra not in palabras_unicas:
                    palabras_unicas.append(palabra)

sentences = np.zeros(((len(normalfiles)+ len(spamfiles)),len(palabras_unicas)+1))

fpath = "normal/"
row = 0
for numero in normalfiles:
    nombre_archivo = fpath + str(numero) + extension
    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip()
                position = palabras_unicas.index(palabra)
                sentences[row,position] = 1
    sentences[row,len(palabras_unicas)] = 0
    row = row + 1

fpath = "spam/"
for numero in spamfiles:
    nombre_archivo = fpath + str(numero) + extension
    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip() 
                position = palabras_unicas.index(palabra)
                sentences[row,position] = 1
    sentences[row,len(palabras_unicas)] = 1
    row = row + 1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_ascent(X, y, theta, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (y - h))
        theta += alpha * gradient
    return theta

theta = np.zeros(sentences.shape[1])
alpha = 0.01  
num_iterations = 1000
sentences = pd.DataFrame(sentences)
sentences.insert(0, 'Ones', 1)

X = sentences.iloc[:, :-1].values
y = sentences.iloc[:, -1].values

optimal_theta = gradient_ascent(X, y, theta, alpha, num_iterations)

fpath = "testing/"
testfiles = np.array([1,2,4,10,11,12,13,18,21,25])

row = 0
sentences_testing = np.zeros((len(testfiles),len(palabras_unicas)))

for numero in testfiles:
    nombre_archivo = fpath + str(numero) + extension
    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip()
                if palabra in palabras_unicas:
                    position = palabras_unicas.index(palabra)
                    sentences_testing[row,position] = 1
    row = row + 1

sentences_testing = pd.DataFrame(sentences_testing)
sentences_testing.insert(0, 'Ones', 1)

logistic_pred = np.dot(sentences_testing, optimal_theta)
y_pred = sigmoid(logistic_pred)

for i in range(len(testfiles)):
    if (y_pred[i] <= 0.5):
        print(testfiles[i],"---","NORMAL")
    else:
        print(testfiles[i],"---","SPAM")

