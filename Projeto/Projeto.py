import pandas as pd
import numpy as np
from simpful import *
import matplotlib.pyplot as plt
FS = FuzzySystem()

df = pd.read_csv('ACI23-24_Proj1_SampleData.csv', sep=',', decimal='.')

Nomes = df.columns
for i, name in enumerate(Nomes[:6]):
    plt.scatter(np.full(df[name].size,i), df[name])
plt.ylim(0, 1.1)
plt.xticks(np.arange(6), Nomes[:6])
plt.show()

CPU1 = TriangleFuzzySet(0, 0, 0.55, term="Low")
CPU2 = TriangleFuzzySet(0.35, 0.5, 0.7, term="Medium")
CPU3 = TrapezoidFuzzySet(0.6, 0.90, 1, 1, term="High")
FS.add_linguistic_variable("ProcessorLoad", LinguisticVariable(
    [CPU1, CPU2, CPU3], universe_of_discourse=[0, 1]))
OutB_1 = TrapezoidFuzzySet(0.5, 0.8, 1, 1, term='High')
OutB_2 = TrapezoidFuzzySet(0, 0, 0.4, 0.7, term='Low')
FS.add_linguistic_variable("OutBandwidth", LinguisticVariable(
    [OutB_1, OutB_2], universe_of_discourse=[0, 1]))


CLP = AutoTriangle(5, terms=["Decreased", "Marginal Decrease", "Maintain",
                   "Marginal Increase", "Increase"], universe_of_discourse=[-1, 1])
FS.add_linguistic_variable("CLPVariation", CLP)


FS.add_rules(["IF ProcessorLoad IS Low THEN CLPVariation IS Increase",
             "IF ProcessorLoad IS Medium THEN CLPVariation IS Maintain", "IF ProcessorLoad IS High THEN CLPVariation IS Decreased"])

xs = []
ys = []
for i in df['ProcessorLoad']:
    print(i, end=", ")
    FS.set_variable("ProcessorLoad", i)
    CLP = FS.inference()['CLPVariation']
    xs.append(i)
    ys.append(CLP)
SSE = np.linalg.norm(ys-df['CLPVariation'])**2
print("\n", df.iloc[:, -1].ravel())
[print(f"{i:.4f}", end=", ") for i in ys]
