import pandas as pd
import numpy as np
from simpful import *
import matplotlib.pylab as plt
from numpy import random

FS_Final = FuzzySystem()
FS_PC_Load = FuzzySystem()
FS_Net_rating = FuzzySystem()


def TabelaDados(iteracoes):
    latency = []
    bandwidth = []
    processor = []
    memory = []
    CLPV = []
    for x in range(iteracoes):
        lat = round(random.uniform(0, 1), 3)
        Bw = round(random.uniform(0, 1), 3)
        proc = round(random.uniform(0, 1), 3)
        mem = round(random.uniform(0, 1), 3)
        latency.append(lat)
        bandwidth.append(Bw)
        memory.append(mem)
        processor.append(proc)
        FS_Net_rating.set_variable("Latency", lat)
        FS_Net_rating.set_variable("Out_BW", Bw)
        Net_infe = FS_Net_rating.inference()['Net_rating']
        FS_PC_Load.set_variable("ProcessorLoad", proc)
        FS_PC_Load.set_variable("MemoryUsage", mem)
        PC_infe = FS_PC_Load.inference()['PC_Load']
        FS_Final.set_variable("PC_Load", PC_infe)
        FS_Final.set_variable("Net_rating", Net_infe)
        CLP_infe = FS_Final.inference()['CLPV']
        CLPV.append(round(CLP_infe, 3))
    df1 = pd.DataFrame(processor, columns=['ProcessorLoad'])
    df1['MemoryUsage'] = memory
    df1['OutBandwidth'] = bandwidth
    df1['Latency'] = latency
    df1['CLPVariation'] = CLPV
    # Dataframe
    df1.to_csv('Train2.csv', sep=',', decimal='.', index=False)


def graficoReal(DIVs):
    Net = []
    PC = []
    PC1 = []
    Net1 = []
    CLPV = []
    # Net_rating
    for lat in np.linspace(0, 1, DIVs):
        for Bw in np.linspace(0, 1, DIVs):
            FS_Net_rating.set_variable("Latency", lat)
            FS_Net_rating.set_variable("Out_BW", Bw)
            Net_infe = FS_Net_rating.inference()['Net_rating']
            Net.append(Net_infe)
    Net = np.array(Net)

    # PC_Load
    for mem in np.linspace(0, 1, DIVs):
        for proc in np.linspace(0, 1, DIVs):
            FS_PC_Load.set_variable("ProcessorLoad", proc)
            FS_PC_Load.set_variable("MemoryUsage", mem)
            PC_infe = FS_PC_Load.inference()['PC_Load']
            PC.append(PC_infe)
    PC = np.array(PC)

    # CLPV
    for x in PC:
        for y in Net:
            FS_Final.set_variable("PC_Load", x)
            FS_Final.set_variable("Net_rating", y)
            CLP_infe = FS_Final.inference()['CLPV']
            CLPV.append(CLP_infe)
            PC1.append(x)
            Net1.append(y)
    CLPV = np.array(CLPV)
    PC1 = np.array(PC1)
    Net1 = np.array(Net1)
    # Figura
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(PC1, Net1, CLPV, vmin=-1, vmax=1, cmap='gnuplot2')
    ax.set_xlabel("PC_Load")
    ax.set_ylabel("Net_rating")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)


def graficoNet_rating(DIVs):
    xs = []
    ys = []
    zs = []
    for x in np.linspace(0, 1, DIVs):
        for y in np.linspace(0, 1, DIVs):
            FS_Net_rating.set_variable("Latency", y)
            FS_Net_rating.set_variable("Out_BW", x)
            CLP = FS_Net_rating.inference()['Net_rating']
            xs.append(x)
            ys.append(y)
            zs.append(CLP)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    ax = fig.add_subplot(221, projection='3d')

    ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=1, cmap='gnuplot2')
    ax.set_ylabel("Latency")
    ax.set_xlabel("Out_BW")
    ax.set_zlim(0, 1)
    return zs.min(), zs.max()


def graficoPC_Load(DIVs):
    xs = []
    ys = []
    zs = []
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

    # fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(223, projection='3d')

    ax.plot_trisurf(xs, ys, zs, vmin=0, vmax=1, cmap='gnuplot2')
    ax.set_xlabel("ProcessorLoad")
    ax.set_ylabel("MemoryUsage")
    ax.set_zlim(0, 1)
    return zs.min(), zs.max()


def graficoFinal(DIVs, PCmin, PCmax, Netmin, Netmax):
    xs = []
    ys = []
    zs = []
    for x in np.linspace(PCmin, PCmax, DIVs):
        for y in np.linspace(Netmin, Netmax, DIVs):
            FS_Final.set_variable("PC_Load", x)
            FS_Final.set_variable("Net_rating", y)
            CLP = FS_Final.inference()['CLPV']
            xs.append(x)
            ys.append(y)
            zs.append(CLP)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_trisurf(xs, ys, zs, vmin=-1, vmax=1, cmap='gnuplot2')
    ax.set_xlabel("PC_Load")
    ax.set_ylabel("Net_rating")
    ax.set_zlim(-1, 1)
    """ plt.tight_layout() """


if __name__ == '__main__':

    df = pd.read_csv('ACI23-24_Proj1_SampleData.csv', sep=',', decimal='.')

    # PC_LOAD ---------------------------------------------------------------------------------------------
    CPU1 = TrapezoidFuzzySet(0, 0, 0.1, 0.55, term="Low")
    CPU2 = TriangleFuzzySet(0.3, 0.6, 0.8, term="Med")
    CPU3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("ProcessorLoad", LinguisticVariable(
        [CPU1, CPU2, CPU3], universe_of_discourse=[0, 1]))

    Memory1 = TrapezoidFuzzySet(0, 0, 0.1, 0.55, term="Low")
    Memory2 = TriangleFuzzySet(0.3, 0.6, 0.8, term="Med")
    Memory3 = TrapezoidFuzzySet(0.6, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("MemoryUsage", LinguisticVariable(
        [Memory1, Memory2, Memory3], universe_of_discourse=[0, 1]))

    PC_Load1 = TrapezoidFuzzySet(0, 0, 0.45, 0.65, term="Low")
    PC_Load2 = TriangleFuzzySet(0.35, 0.7, 0.85, term="Med")
    PC_Load3 = TrapezoidFuzzySet(0.7, 0.85, 1, 1, term="High")
    FS_PC_Load.add_linguistic_variable("PC_Load", LinguisticVariable(
        [PC_Load1, PC_Load2, PC_Load3], universe_of_discourse=[0, 1]))

    FS_PC_Load.add_rules(["IF (MemoryUsage IS High) AND (ProcessorLoad IS High) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS High) AND (ProcessorLoad IS Low) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS High) AND (ProcessorLoad IS Med) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS Med) AND (ProcessorLoad IS High) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Med) THEN (PC_Load IS Med)",
                          "IF (MemoryUsage IS Med) AND (ProcessorLoad IS Low) THEN (PC_Load IS Low)",
                          "IF (MemoryUsage IS Low) AND (ProcessorLoad IS High) THEN (PC_Load IS High)",
                          "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Med) THEN (PC_Load IS Low)",
                          "IF (MemoryUsage IS Low) AND (ProcessorLoad IS Low) THEN (PC_Load IS Low)",
                          ])

    # Net_rating ---------------------------------------------------------------------------------------------
    Latency1 = TrapezoidFuzzySet(0, 0, 0.3, 0.7, term="Low")
    Latency2 = TrapezoidFuzzySet(0.4, 0.8, 1, 1, term="High")
    FS_Net_rating.add_linguistic_variable("Latency", LinguisticVariable(
        [Latency1, Latency2], universe_of_discourse=[0, 1]))

    Out_BW1 = TrapezoidFuzzySet(0, 0, 0.2, 0.45, term="Low")
    Out_BW2 = TrapezoidFuzzySet(0.2, 0.35, 0.55, 0.7, term="Med")
    Out_BW3 = TrapezoidFuzzySet(0.45, 0.7, 1, 1, term="High")
    FS_Net_rating.add_linguistic_variable("Out_BW", LinguisticVariable(
        [Out_BW1, Out_BW2, Out_BW3], universe_of_discourse=[0, 1]))

    Net_rating1 = TrapezoidFuzzySet(0, 0, 0.2, 0.3, term="Bad")
    Net_rating2 = TriangleFuzzySet(0.1, 0.3, 0.55, term="Mid")
    Net_rating3 = TrapezoidFuzzySet(0.35, 0.5, 1, 1, term="Good")
    FS_Net_rating.add_linguistic_variable("Net_rating", LinguisticVariable(
        [Net_rating1, Net_rating2, Net_rating3], universe_of_discourse=[0, 1]))

    FS_Net_rating.add_rules(["IF (Out_BW IS Low) THEN (Net_rating IS Bad)",
                             "IF (Out_BW IS Med) AND (Latency IS Low) THEN (Net_rating IS Mid)",
                             "IF (Out_BW IS Med) AND (Latency IS High) THEN (Net_rating IS Mid)",
                             "IF (Latency IS High) AND (Out_BW IS High) THEN (Net_rating IS Mid)",
                             "IF (Latency IS Low) AND (Out_BW IS High) THEN (Net_rating IS Good)"])

    # FS_Net_rating.set_variable("Latency", df["Latency"][i])
    # FS_Net_rating.set_variable("Out_BW", df["OutBandwidth"][i])
    # Net_rating_ = FS_Net_rating.inference(['Net_rating'])
    # # print(Net_rating_)

    # CLPV -------------------------------------------------------------------------------------------------
    PC_Load2 = TriangleFuzzySet(0.45, 0.675, 0.85, term="Med")
    FS_Final.add_linguistic_variable("PC_Load", LinguisticVariable(
        [PC_Load1, PC_Load2, PC_Load3], universe_of_discourse=[0, 1]))
    FS_Final.add_linguistic_variable("Net_rating", LinguisticVariable(
        [Net_rating1, Net_rating2, Net_rating3], universe_of_discourse=[0, 1]))

    CLPV_D = TrapezoidFuzzySet(-1, -1, -0.65, -0.425, term="Decrease")
    CLPV_MD = TrapezoidFuzzySet(-0.7, -0.3, -0.2, 0.2, term="MDecrease")
    CLPV_MI = TrapezoidFuzzySet(-0.1, 0, 0.1, 0.7, term="MIncrease")
    CLPV_I = TrapezoidFuzzySet(0.425, 0.65, 1, 1, term="Increase")
    FS_Final.add_linguistic_variable("CLPV", LinguisticVariable(
        [CLPV_D, CLPV_MD, CLPV_MI, CLPV_I], universe_of_discourse=[-1, 1]))

    FS_Final.add_rules(["IF (PC_Load IS Low) AND (Net_rating IS Bad) THEN (CLPV IS Increase)",
                        "IF (PC_Load IS Low) AND (Net_rating IS Mid) THEN (CLPV IS Increase)",
                        "IF (PC_Load IS Low) AND (Net_rating IS Good) THEN (CLPV IS Increase)",
                        "IF (PC_Load IS Med) AND (Net_rating IS Bad) THEN (CLPV IS Increase)",
                        "IF (PC_Load IS Med) AND (Net_rating IS Mid) THEN (CLPV IS Increase)",
                        "IF (PC_Load IS Med) AND (Net_rating IS Good) THEN (CLPV IS MIncrease)",
                        "IF (PC_Load IS High) AND (Net_rating IS Bad) THEN (CLPV IS MDecrease)",
                        "IF (PC_Load IS High) AND (Net_rating IS Mid) THEN (CLPV IS Decrease)",
                        "IF (PC_Load IS High) AND (Net_rating IS Good) THEN (CLPV IS Decrease)"
                        ])

    # Testes -------------------------------------------------------------------------------------------------
    PC_Load = np.array([])
    Net_rating = np.array([])
    CLPV = np.array([])

    for i in range(df.shape[0]):
        FS_PC_Load.set_variable("ProcessorLoad", df["ProcessorLoad"][i])
        FS_PC_Load.set_variable("MemoryUsage", df["MemoryUsage"][i])
        PC_Load_ = FS_PC_Load.inference()['PC_Load']

        PC_Load = np.append(PC_Load, PC_Load_)
        FS_Net_rating.set_variable("Latency", df["Latency"][i])
        FS_Net_rating.set_variable("Out_BW", df["OutBandwidth"][i])
        Net_rating_ = FS_Net_rating.inference()['Net_rating']

        Net_rating = np.append(Net_rating, Net_rating_)

        FS_Final.set_variable("PC_Load", PC_Load_)
        FS_Final.set_variable("Net_rating", Net_rating_)
        CLPV_ = FS_Final.inference()['CLPV']
        CLPV = np.append(CLPV, CLPV_)

    print('PC_Load =\n', PC_Load.ravel())
    print('Net_rating =\n', Net_rating.ravel())
    print('CLPV =\n', CLPV.ravel())
    valores = df[['CLPVariation']].to_numpy() - CLPV.reshape(-1, 1)
    diference = abs(df[['CLPVariation']].to_numpy() - CLPV.reshape(-1, 1))
    print('diff(%) =')
    [print(f'{i:.2f} em index {idx}')
     for idx, i in enumerate((valores*100).ravel())]
    print('Soma do erro=', round(np.sum(diference), 3),
          '\nStandard Deviation%', round(np.std(diference)*100, 3))

    # Graficos

    """ fig = plt.figure(figsize=(17, 8))
    Net_min, Net_max = graficoNet_rating(21)
    PC_min, Pc_max = graficoPC_Load(21)
    graficoFinal(21, PC_min, Pc_max, Net_min, Net_max) """
    """ TabelaDados(4000) """
    """ plt.savefig("PDFs/graficos.pdf")
    plt.tight_layout()
    plt.show()"""
    FS_Final.produce_figure("PDFs/Final.pdf")
    FS_PC_Load.produce_figure("PDFs/PC_Load.pdf")
    FS_Net_rating.produce_figure("PDFs/Net_rating.pdf")
