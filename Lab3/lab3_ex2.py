import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def variacao(precos):
    flag = -1
    valores = np.zeros(len(precos)-1)
    for i, row in enumerate(precos, -1):
        if flag != -1:
            valores[i] = row - flag
        flag = row
    return valores


def highslows(High, Low):
    valores = np.zeros(len(High))
    for i, [high, low] in enumerate(zip(High, Low)):
        valores[i] = high - low
    return valores


# Ficheiro do enunciado original
df = pd.read_csv("DCOILBRENTEUv2.csv", sep=",", decimal=".")
valores = variacao(df["DCOILBRENTEU"])
plt.hist(valores, bins=50)
plt.title("Variação do preço do óleo")


# Com o outro ficheiro
df = pd.read_csv("Teste1.csv", sep=";", decimal=",")
var = variacao(df["Close"])
HighLow = highslows(df["High"], df["Low"])
plt.figure()
plt.subplot(2, 1, 1)
plt.title("Variation between days")
print(var)
plt.hist(var, bins=30)
plt.subplot(2, 1, 2)
plt.hist(HighLow, bins=30)
plt.title("High-Low")
plt.show()
