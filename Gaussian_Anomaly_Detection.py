import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get Mean and STD from training dataset
data_t = pd.read_csv("../train/train/training-data.csv")

data_t1 = data_t.iloc[:, [1,2,4,5,6,8,9,10,12,13,14,16,18,20]]

mean_t = data_t1.mean(axis=0)
std_t = data_t1.std(axis=0)

data_t_scaled = (data_t1 - mean_t)/std_t

mean = data_t_scaled.mean()
var = data_t_scaled.var()

valtst = input("Enter v for (v)alidation or t for (t)est: ")
ext = ".csv"

if  (valtst == "v"):
    finrango = 23
    fpath = "../validation/validation/"
elif (valtst == "t"):
    finrango = 58
    fpath = "../test/test/"
else:
    finrango = 0

for fname in range(0,finrango):
    # Readfile  CSV
    filename = str(fname)
    file = fpath + filename + ext
    print(file)

    data = pd.read_csv(file)
    data1 = data.iloc[:, [1,2,4,5,6,8,10,12,13,14,16,18,20]]

    data_scaled = (data1 - mean_t)/std_t

    p_x = np.exp(-0.5*np.square((data_scaled - mean))/var)/np.sqrt(2*np.pi*var)

    p_x = np.prod(p_x,axis=1)

    # Find umbral
    #frepx = p_x.value_counts()
    #for valor, frecuencia in frepx.items():
    #    print(f'Valor: {valor}, Frecuencia: {frecuencia}')
    #input("frecuencias")
    """
    umbral=0.01
    intervalo = 0.01
    for j in range (0,300):
        c=0
        for i in range(0,len(p_x)):
            if  (p_x[i]>umbral):
                c=c+1
        print(j,umbral,c)
        if  (c<=5800):
            umbral = umbral - intervalo
            intervalo = intervalo - intervalo / 5
            print("intervalo",intervalo)
        umbral = umbral + intervalo
        #input("umbral")

    #input("key")
    """
    c=0
    #umbral = 0.000000023509  #para 7000
    #umbral = 0.000000055372 #para 6000
    umbral = 0.000000068429 #para 5800
    for i in range(0,len(p_x)):
        if  (p_x[i]>umbral):
            c=c+1
    print("Number of normal rows in the file", c) 
    if (c < 5800):
        print("Abnormal File")
    else:
        print("Normal File")

