import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the mean and std from training  file
data_t = pd.read_csv("../train/train/training-data.csv")
data_t1 = data_t.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
mean_t = data_t1.mean(axis=0)
std_t = data_t1.std(axis=0)

# Normalization
data_t_scaled = (data_t1 - mean_t)/std_t
mean = data_t_scaled.mean()
covar = data_t_scaled.cov()



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
    data1 = data.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    data_scaled = (data1 - mean_t)/std_t

    dim = len(mean)
    cn = 1 / ((2 * np.pi) ** (dim / 2) * np.linalg.det(covar) ** 0.5)

    p_x = []
    mean = np.array(mean)
    data_scaled = np.array(data_scaled)

    for row in data_scaled:
        calc1 = row - mean
        expo = -0.5 * np.dot(np.dot(calc1.T, np.linalg.inv(covar)), calc1)
        prob = np.exp(expo) * cn
        p_x.append(prob)


    # Find umbral
    #frepx = p_x.value_counts()
    #for valor, frecuencia in frepx.items():
    #    print(f'Valor: {valor}, Frecuencia: {frecuencia}')
    #input("frecuencias")
    """
    umbral=5
    intervalo = 10
    for j in range (0,300):
        c=0
        for i in range(0,len(p_x)):
            if  (p_x[i]>umbral):
                c=c+1
        print(j,umbral,c)
        if  (c<=9000):
            umbral = umbral - intervalo
            intervalo = intervalo - intervalo / 5
            print("intervalo",intervalo)
        umbral = umbral + intervalo
        #input("umbral")

    #input("key")
    """


    c=0
    umbral = 0.0000140000  #para 9000
    #umbral = 0.0019458 #para 8500
    for i in range(0,len(p_x)):
        if  (p_x[i]>umbral):
            c=c+1
    print("Number of normal rows in the file", c) 
    if (c < 9000):
        print("Abnormal File")
    else:
        print("Normal File")
