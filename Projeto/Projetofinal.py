import pandas as pd
import numpy as np
from simpful import *
import matplotlib.pylab as plt

FS_Final = FuzzySystem()
FS_PC_Load = FuzzySystem()
FS_Net_Load = FuzzySystem()
FS_Net_Trans = FuzzySystem()
FS_Net_Free = FuzzySystem()


if __name__ == '__main__':

    df = pd.read_csv('ACI23-24_Proj1_SampleData.csv', sep=',', decimal='.')
    """ Lat_1 = TrapezoidFuzzySet(0.4, 0.7, 1, 1, term='High')
    Lat_2 = TrapezoidFuzzySet(0, 0, 0.3, 0.6, term='Low')
    FS_Net_Free.add_linguistic_variable("Latency", LinguisticVariable(
        [Lat_1, Lat_2], universe_of_discourse=[0, 1]))
    OutB_1 = TrapezoidFuzzySet(0.5, 0.8, 1, 1, term='High')
    OutB_2 = TrapezoidFuzzySet(0, 0, 0.4, 0.7, term='Low')
    FS_Net_Free.add_linguistic_variable("OutBandwidth", LinguisticVariable(
        [OutB_1, OutB_2], universe_of_discourse=[0, 1]))
    Fuzzy1 = AutoTriangle(
        3, terms=["Low", "Medim", "High"], universe_of_discourse=[0, 1])
    FS_Net_Free.add_linguistic_variable("Fuzzy1", Fuzzy1) """

    CPU1 = TriangleFuzzySet(0, 0, 0.55, term="Low")
    CPU2 = TriangleFuzzySet(0.35, 0.5, 0.7, term="Medium")
    CPU3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("ProcessorLoad", LinguisticVariable(
        [CPU1, CPU2, CPU3], universe_of_discourse=[0, 1]))
    Memory1 = TriangleFuzzySet(0, 0, 0.55, term="Low")
    Memory2 = TriangleFuzzySet(0.35, 0.5, 0.7, term="Medium")
    Memory3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("MemoryUsage", LinguisticVariable(
        [Memory1, Memory2, Memory3], universe_of_discourse=[0, 1]))
    PC_Load1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
    PC_Load2 = TriangleFuzzySet(0.4, 0.7, 0.85, term="Med")
    PC_Load3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("PC_Load", LinguisticVariable(
        [PC_Load1, PC_Load2, PC_Load3], universe_of_discourse=[0, 1]))
    """ FS_PC_Load.produce_figure() """

    FS_PC_Load.add_rules(["IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS High) AND (NOT( ProcessorLoad IS High)) THEN (PC_Load IS Med)",
                          "IF (MemoryUsage IS Medium) AND (ProcessorLoad IS High) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS Medium) AND (ProcessorLoad IS Medium) THEN (PC_Load IS Med)",
                          "IF (MemoryUsage IS Medium) AND (ProcessorLoad IS Low) THEN (PC_Load IS Low)",
                          "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (PC_Load IS Med)",
                          "IF (MemoryUsage IS Low) AND (NOT(ProcessorLoad IS High)) THEN (PC_Load IS Low)",
                          ])



    Fuzzy_Net_Trans1 = TrapezoidFuzzySet(0,0 ,0.3, 0.6,term="Received")
    Fuzzy_Net_Trans2 = TrapezoidFuzzySet(0.4, 0.7, 1, 1,term="Transmitted")
    FS_Net_Load.add_linguistic_variable("Net_Trans", LinguisticVariable(
        [Fuzzy_Net_Trans1, Fuzzy_Net_Trans2], universe_of_discourse=[0, 1]))
    FS_Net_Load.produce_figure()
    Fuzzy_Net_free1 = TrapezoidFuzzySet(0,0 ,0.3, 0.6,term="Free")
    Fuzzy_Net_free2 = TrapezoidFuzzySet(0.4, 0.7, 1, 1,term="Busy")
    FS_Net_Load.add_linguistic_variable("Net_free", LinguisticVariable([Fuzzy_Net_free1, Fuzzy_Net_free2], universe_of_discourse=[0, 1]))
    Net_Load1 = TrapezoidFuzzySet(0, 0, 0.3, 0.5, term="Low")
    Net_Load2 = TriangleFuzzySet(0.4, 0.7, 0.85, term="Med")
    Net_Load3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_Net_Load.add_linguistic_variable("Net_free", LinguisticVariable([Fuzzy_Net_free1, Fuzzy_Net_free2], universe_of_discourse=[0, 1]))


""" xs = []
ys = []
zs = []
DIVs = 21
for x in np.linspace(0, 1, DIVs):
    for y in np.linspace(0, 1, DIVs):
        FS_PC_Load.set_variable("ProcessorLoad", x)
        FS_PC_Load.set_variable("MemoryUsage", y)
        CLP = FS_PC_Load.inference()['PC_Load']
        xs.append(x)
        ys.append(y)
        zs.append(CLP)
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111, projection='3d')

xx, yy = plt.meshgrid(xs, ys)

ax.plot_trisurf(xs, ys, zs, vmin=-1, vmax=1, cmap='gnuplot2')
ax.set_xlabel("Proc")
ax.set_zlim(-1, 1)
plt.tight_layout()
plt.show() """
