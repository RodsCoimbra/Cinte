import pandas as pd
import numpy as np
from simpful import *
import matplotlib.pylab as plt
from numpy import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor

df = pd.read_csv('Train.csv')
colunas = df.columns
x = df[colunas[0:4]]
y = df[colunas[4]]
x_t = x.to_numpy()
y_t = y.to_numpy()
rand = random.randint(0, 1000000)

base1 = MLPRegressor(hidden_layer_sizes=(3), max_iter=1000, activation='logistic',
                     solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

base2 = MLPRegressor(hidden_layer_sizes=(3), max_iter=1000, activation='relu',
                     solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

base3 = MLPRegressor(hidden_layer_sizes=(3), max_iter=1000, activation='tanh',
                     solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

base = [base1, base2, base3]

hidden1 = MLPRegressor(hidden_layer_sizes=(4), max_iter=1000, activation='logistic',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden2 = MLPRegressor(hidden_layer_sizes=(4), max_iter=1000, activation='relu',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden3 = MLPRegressor(hidden_layer_sizes=(4), max_iter=1000, activation='tanh',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden = [hidden1, hidden2, hidden3]

hidden4 = MLPRegressor(hidden_layer_sizes=(5), max_iter=1000, activation='logistic',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden5 = MLPRegressor(hidden_layer_sizes=(5), max_iter=1000, activation='relu',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden6 = MLPRegressor(hidden_layer_sizes=(5), max_iter=1000, activation='tanh',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

final = [hidden4, hidden5, hidden6]


hidden7 = MLPRegressor(hidden_layer_sizes=(6), max_iter=1000, activation='logistic',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden8 = MLPRegressor(hidden_layer_sizes=(6), max_iter=1000, activation='relu',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

hidden9 = MLPRegressor(hidden_layer_sizes=(6), max_iter=1000, activation='tanh',
                       solver='sgd', learning_rate='adaptive', learning_rate_init=0.01)

final2 = [hidden7, hidden8, hidden9]
regressoes = [base, hidden, final, final2]
""" x_train, x_test, y_train, y_test = train_test_split(
    x_t, y_t, test_size=0.2, random_state=rand) """
kf = KFold(n_splits=10, random_state=rand, shuffle=True)
for idx, rel in enumerate(regressoes):
    print("\n\n================================\n", idx)
    for reg in rel:
        SSE2 = 0
        Max = []
        for idx_train, idx_teste in kf.split(x_t):
            reg.fit(x_t[idx_train], y_t[idx_train])
            y_prev = reg.predict(x_t[idx_teste])
            SSE = np.linalg.norm(y_prev - y_t[idx_teste])**2
            Max = np.append(Max, SSE)
            SSE2 += SSE
        print("--------------------------------\n", reg)
        print("SSE: ", SSE2/kf.get_n_splits())
        print("Max:", Max.max())
