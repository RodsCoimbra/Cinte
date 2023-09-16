import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime


def find_outliers(Value, k):
    return np.where(abs(Value.mean() - k * Value.std()) < abs(Value.mean() - Value))


def remove_row_csv(i, df_orig):
    df = df_orig.drop(i[0])
    return df


def change_row_to_previous(i, df_orig):
    df = df_orig.copy()
    for a in i:
        df.iloc[a[0], 1:] = df.iloc[a[0]-1, 1:]
    return df


def interpolate_rows(i, df_orig):
    df = df_orig.copy()
    for a in i:
        df.iloc[a[0], 1:] = (df.iloc[a[0]-1, 1:] + df.iloc[a[0]+1, 1:])/2
    return df


# Main
df = pd.read_csv("EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv",
                 sep=";", decimal=",")

# FUNÇÕES
indexs_outliers = find_outliers(df['High'], 3)
df1 = remove_row_csv(indexs_outliers, df)
df2 = change_row_to_previous(indexs_outliers, df)
df3 = interpolate_rows(indexs_outliers, df)

# CSV PARA VISUALIZAR
df1.to_csv("Teste1.csv", index=False, sep=";", decimal=",")
df2.to_csv("Teste2.csv", index=False, sep=";", decimal=",")
df3.to_csv("Teste3.csv", index=False, sep=";", decimal=",")

# PLOTS
plt.figure()
Dataframes = [df, df1, df2, df3]
for idx, dfs in enumerate(Dataframes, 1):
    plt.subplot(2, 2, idx)
    plt.plot(pd.to_datetime(dfs['Time (UTC)']
                            ).to_numpy(), dfs['High'].to_numpy())
    plt.title("Gráfico: " + str(idx))
plt.show()
