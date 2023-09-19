import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


col = ['s_len', 's_wid', 'p_len', 'p_wid', 'class']

df = pd.read_csv("iris/iris.data", header=None,
                 names=col, sep=",", decimal=".")
""" sns.relplot(x=df[col[0]], y=df[col[1]], size=df[col[3]],
            hue=df[col[4]], style=df[col[4]])
plt.show() """

x_train, x_test, y_train, y_test = train_test_split(df[col[0:4]], df[col[4]], test_size=0.3, random_state=3)

clf = GaussianNB()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(accuracy_score(y_true=y_test, y_pred=y_predict))

