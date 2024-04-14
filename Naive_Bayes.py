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
                sentences[row,position] = sentences[row,position] + 1
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
                sentences[row,position] = sentences[row,position] + 1
    sentences[row,len(palabras_unicas)] = 1
    row = row + 1

condprob = np.zeros((2,len(palabras_unicas)))

normalprobtot = len(normalfiles)
spamprobtot = len(spamfiles)

for i in range(len(palabras_unicas)):
    normalprob = 0
    for j in range(len(normalfiles)+len(spamfiles)):
        if (sentences[j,len(palabras_unicas)] == 0):
            normalprob = normalprob + sentences[j,i]
    condprob[0,i] = normalprob / normalprobtot

for i in range(len(palabras_unicas)):
    spamprob = 0
    for j in range(len(normalfiles)+len(spamfiles)):
        if (sentences[j,len(palabras_unicas)] == 1):
            spamprob = spamprob + sentences[j,i]
    condprob[1,i] = spamprob / spamprobtot


fpath = "testing/"
testfiles = np.array([1,2,4,10,11,12,13,18,21,25])

for numero in testfiles:
    nombre_archivo = fpath + str(numero) + extension
    probnormal = normalprobtot / (normalprobtot + spamprobtot)
    probspam = spamprobtot / (normalprobtot + spamprobtot)

    with open(nombre_archivo, "r") as archivo:
        for linea in archivo:
            palabras = linea.split()
            for palabra in palabras:
                palabra = re.sub(r'[^A-Za-z0-9]', '', palabra)
                palabra = palabra.strip()  
                if palabra in palabras_unicas:
                    position = palabras_unicas.index(palabra)
                    if (condprob[0,position] == 0):
                        probnormal = probnormal / len(palabras_unicas)
                    else:
                        probnormal = probnormal * condprob[0,position]
                    if (condprob[1,position] == 0):
                        probspam = probspam / len(palabras_unicas)
                    else:
                         probspam = probspam * condprob[1,position]

    if (probnormal > probspam):
       print(nombre_archivo,"---","NORMAL")
    else:
       print(nombre_archivo,"---","SPAM")
