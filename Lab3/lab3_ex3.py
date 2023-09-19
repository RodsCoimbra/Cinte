import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df1 = pd.read_csv("DCOILBRENTEUv2.csv", sep=",", decimal=".")
df2 = pd.read_csv("DCOILWTICOv2.csv", sep=",", decimal=".")
plt.subplot(2, 1, 1)
plt.scatter(pd.to_datetime(df1["DATE"]).to_numpy(),
         df1["DCOILBRENTEU"].to_numpy())
plt.subplot(2, 1, 2)
plt.scatter(pd.to_datetime(df2["DATE"]).to_numpy(), df2["DCOILWTICO"].to_numpy())

""" plt.figure()
plt.scatter(df1["DCOILBRENTEU"].to_numpy(), df2["DCOILWTICO"].to_numpy())
plt.xlabel("Price on UK")
plt.ylabel("Price on Texas")
plt.title("Price of oil") """
plt.show()
