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
col2 = ['Age', 'Year', 'Nodes', 'Survival']
df = pd.read_csv("iris/iris.data", header=None,
                 names=col, sep=",", decimal=".")
df2 = pd.read_csv("haberman/haberman.data", header=None, sep=",",
                  decimal=".", names=col2)
""" sns.relplot(x=df[col[0]], y=df[col[1]], size=df[col[3]],
            hue=df[col[4]], style=df[col[4]])
plt.show() """

x_train, x_test, y_train, y_test = train_test_split(
    df[col[0:4]], df[col[4]], test_size=0.3, random_state=3)
x2_train, x2_test, y2_train, y2_test = train_test_split(
    df2[col2[0:3]], df2[col2[3]], test_size=0.3, random_state=3)

guassian = GaussianNB()
guassian.fit(x_train, y_train)
y_predict = guassian.predict(x_test)
guassian2 = GaussianNB()
guassian2.fit(x2_train, y2_train)
y_predict2 = guassian2.predict(x2_test)
acc = accuracy_score(y_true=y_test, y_pred=y_predict)
prec = precision_score(y_true=y_test, y_pred=y_predict, average='macro')
recall = recall_score(y_true=y_test, y_pred=y_predict, average='macro')
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print("Guassian Values1:", acc, prec, recall, "\n", conf_mat, "\n\n")
acc = accuracy_score(y_true=y2_test, y_pred=y_predict2)
prec = precision_score(y_true=y2_test, y_pred=y_predict2, average='macro')
recall = recall_score(y_true=y2_test, y_pred=y_predict2, average='macro')
conf_mat = confusion_matrix(y_true=y2_test, y_pred=y_predict2)
print("Guassian Values2:", acc, prec, recall, "\n", conf_mat, "\n\n")

""" linearSVC = LinearSVC()
linearSVC.fit(x_train, y_train)
y_predict = linearSVC.predict(x_test)
linearSVC2 = LinearSVC()
linearSVC2.fit(x2_train, y2_train)
y_predict2 = linearSVC2.predict(x2_test)

acc = accuracy_score(y_true=y_test, y_pred=y_predict)
prec = precision_score(y_true=y_test, y_pred=y_predict, average='macro')
recall = recall_score(y_true=y_test, y_pred=y_predict, average='macro')
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print("linearSVC Values1:", acc, prec, recall, "\n", conf_mat, "\n\n")
acc = accuracy_score(y_true=y2_test, y_pred=y_predict2)
prec = precision_score(y_true=y2_test, y_pred=y_predict2, average='macro')
recall = recall_score(y_true=y2_test, y_pred=y_predict2, average='macro')
conf_mat = confusion_matrix(y_true=y2_test, y_pred=y_predict2)
print("linearSVC Values2:", acc, prec, recall, "\n", conf_mat, "\n\n")

SVM = svm()
SVM.fit(x_train, y_train)
y_predict = SVM.predict(x_test)
SVM2 = svm()
SVM2.fit(x2_train, y2_train)
y_predict2 = SVM2.predict(x2_test)

acc = accuracy_score(y_true=y_test, y_pred=y_predict)
prec = precision_score(y_true=y_test, y_pred=y_predict, average='macro')
recall = recall_score(y_true=y_test, y_pred=y_predict, average='macro')
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print("SVM Values1:", acc, prec, recall, "\n", conf_mat, "\n\n")
acc = accuracy_score(y_true=y2_test, y_pred=y_predict2)
prec = precision_score(y_true=y2_test, y_pred=y_predict2, average='macro')
recall = recall_score(y_true=y2_test, y_pred=y_predict2, average='macro')
conf_mat = confusion_matrix(y_true=y2_test, y_pred=y_predict2)
print("SVM Values2:", acc, prec, recall, "\n", conf_mat, "\n\n")

Neighbors = KNeighborsClassifier()
Neighbors.fit(x_train, y_train)
y_predict = Neighbors.predict(x_test)
Neighbors2 = KNeighborsClassifier()
Neighbors2.fit(x2_train, y2_train)
y_predict2 = Neighbors2.predict(x2_test)

acc = accuracy_score(y_true=y_test, y_pred=y_predict)
prec = precision_score(y_true=y_test, y_pred=y_predict, average='macro')
recall = recall_score(y_true=y_test, y_pred=y_predict, average='macro')
conf_mat = confusion_matrix(y_true=y_test, y_pred=y_predict)
print("KNeighborsClassifier Values1:", acc, prec, recall, "\n", conf_mat, "\n\n")
acc = accuracy_score(y_true=y2_test, y_pred=y_predict2)
prec = precision_score(y_true=y2_test, y_pred=y_predict2, average='macro')
recall = recall_score(y_true=y2_test, y_pred=y_predict2, average='macro')
conf_mat = confusion_matrix(y_true=y2_test, y_pred=y_predict2)
print("KNeighborsClassifier Values2:", acc, prec, recall, "\n", conf_mat, "\n\n")"""
