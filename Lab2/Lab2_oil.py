import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Lab2/DCOILBRENTEUv2.csv")
df1 = df['DCOILBRENTEU']
min = df1.min()
max = df1.max()
df_novo = pd.DataFrame()
df_novo['Normalize'] = (df1 - min)/(max-min)
miu = df1.mean()
omega = df1.std()
df_novo['Standardize'] = (df1 - miu)/omega
df_novo.to_csv("Ver.csv", index = False)
plt.subplot(211)
plt.plot(df_novo['Normalize'], linewidth=2)
plt.xlim(0, len(df))
plt.grid()
plt.subplot(212)
plt.plot(df_novo['Standardize'], 'r', linewidth=2)
plt.xlim(0, len(df))
plt.grid()
plt.show()
