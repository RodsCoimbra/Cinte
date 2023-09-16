import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("DCOILBRENTEUv2.csv", sep = ",", decimal= ".")
plt.hist(pd.to_datetime(df["DATE"]).to_numpy(), df["DCOILBRENTEU"].to_numpy())
plt.show()