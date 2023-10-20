import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools
from sys import exit
from icecream import ic
import random
def ROI_total(Roi_short_sum, Roi_long_sum):
    return (Roi_short_sum + Roi_long_sum) / 2


def ROI_long(sell, buy):
    return ((sell - buy) / buy) * 100


def ROI_short(sell, buy):
    return ((sell - buy) / sell) * 100


def RSI(df):
    df['RSI_7'] = 100 - (100 / (1 +RS(df['Gain'].rolling(7).mean(),
                    df['Loss'].rolling(7).mean())))
    df['RSI_14'] = 100 - (100 / (1 +RS(df['Gain'].rolling(14).mean(),
                     df['Loss'].rolling(14).mean())))
    df['RSI_21'] = 100 - (100 / (1 +RS(df['Gain'].rolling(21).mean(),
                     df['Loss'].rolling(21).mean())))
    return df


def RS(avg_gain, avg_loss):
    return avg_gain / avg_loss


def ROI_results(RSI_short, RSI_long, lll, ull, lls, uls):
    # if (RSI_long != 0 and RSI_short != 0 and RSI_short % 7 != 0 and RSI_long % 7 != 0 and RSI_short > 21 and RSI_long > 21
    #         and lll % 5 != 0 and ull % 5 != 0 and lls % 5 != 0 and uls % 5 != 0
    #         and lll > 100 and ull > 100 and lls > 100 and uls > 100):
    #     print("ERRO, parametro mal")
    #     return
    RSI_period_long = 'RSI_' + str(RSI_long)
    RSI_period_short = 'RSI_' + str(RSI_short)
    flag_short = False
    flag_long = False
    Roi_short = []
    Roi_long = []
    sell_short = 0
    buy_long = 0
    for idx, [valueshort, valuelong] in enumerate(zip(df[RSI_period_short], df[RSI_period_long])):
        if (df[RSI_period_short].size == idx + 2):
            if (flag_short):
                buy_short = df['Close'].iloc[idx+1]
                flag_short = False
                Roi_short = np.append(
                    Roi_short, ROI_short(sell_short, buy_short))
            if (flag_long):
                sell_long = df['Close'].iloc[idx+1]
                flag_long = False
                Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
            break

        else:
            if (valueshort > uls and df[RSI_period_short].iloc[idx+1] < uls and flag_short == False):
                sell_short = df['Close'].iloc[idx+1]
                flag_short = True
            elif (valueshort > lls and df[RSI_period_short].iloc[idx+1] < lls and flag_short == True):
                buy_short = df['Close'].iloc[idx+1]
                flag_short = False
                Roi_short = np.append(
                    Roi_short, ROI_short(sell_short, buy_short))
            if (valuelong < lll and df[RSI_period_long].iloc[idx+1] > lll and flag_long == False):
                buy_long = df['Close'].iloc[idx+1]
                flag_long = True
            elif (valuelong < ull and df[RSI_period_long].iloc[idx+1] > ull and flag_long == True):
                sell_long = df['Close'].iloc[idx+1]
                flag_long = False
                Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
    return ROI_total(np.sum(Roi_short), np.sum(Roi_long))

def evaluate(individual):
    genes = np.zeros(6, dtype=int)
    for idx, vars in enumerate(individual):
        genes[idx] = vars
    # if (genes[2]<=20 and genes[2]>=0 and genes[3]<=20 and genes[3]>=0 and genes[4]<=20 and genes[4]>=0 and genes[5]<=20 and genes[5]>=0 and
    #      genes[0] >= 1 and genes[0] <= 3 and genes[1] >= 1 and genes[1] <= 3):
    genes[0] *= 7
    genes[1] *= 7
    genes[2] *= 5
    genes[3] *= 5
    genes[4] *= 5
    genes[5] *= 5
    return ROI_results(genes[0], genes[1], genes[2], genes[3], genes[4], genes[5]),
    # else:
    #     for i in genes:
    #         print(i)
    #     print("ERRO, parametro mal")
    #     exit(0)

def create_EA():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("RSI_short", random.randint, 1, 3)
    toolbox.register("RSI_long", random.randint, 1, 3)
    toolbox.register("lll", random.randint, 0, 20)
    toolbox.register("ull", random.randint, 0, 20)
    toolbox.register("lls", random.randint, 0, 20)
    toolbox.register("uls", random.randint, 0, 20)
    toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.RSI_short, toolbox.RSI_long, toolbox.lll, toolbox.ull, toolbox.lls, toolbox.uls), n = 1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)

    toolbox.register("mate", tools.cxTwoPoint)

    toolbox.register("mutate", tools.mutUniformInt, low= 0,up= 20, indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def EA(toolbox):
    

    pop = toolbox.population(n=100)

    CXPB, MUTPB = 0.6, 0.35

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0

    while g < 100:
        # A new generation
        Max_early = -1000
        g_early = 0
        g = g + 1
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                if (mutant[0] < 1 or mutant[0]>3):
                    mutant[0]%= 3
                    mutant[0]+=1
                if (mutant[1] < 1 or mutant[1]>3):
                    mutant[1]%=3
                    mutant[1]+=1
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        if(g%20==0):
            print("-- Generation %i --" % g)
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
        if (Max_early < max(fits)):
            Max_early = max(fits)
            g_early = g
        if (g - g_early > 15):
            print("\n-------------------------------Early stop---------------------------")
            break
    
    best_ind = tools.selBest(pop, 1)[0]
    best_genes = np.array(best_ind)

    print("Best individual is",best_genes[0:2] * 7,best_genes[2:]*5, best_ind.fitness.values)
    return np.append (best_genes[0:2] * 7, best_genes[2:]*5), best_ind.fitness.values[0]





if __name__ == '__main__':
    path = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F',
            'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']
    df = {}

    toolbox = create_EA()
    # Read data from csv files
    start = pd.to_datetime('01-01-2020', dayfirst=True)
    end = pd.to_datetime('31-12-2022', dayfirst=True)
    # start = pd.to_datetime('01-08-2023', dayfirst=True)
    # end = pd.to_datetime('15-09-2023', dayfirst=True)
     
    for i in path:
        print("\n\n\n--------------Path ", i, "-------------------------\n\n")
        df = pd.read_csv('Data/' + i + '.csv', sep=';', decimal='.',
                            usecols=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df[(df['Date'] >= start) & (df['Date'] <= end)]
        diff = pd.Series(df['Close'].diff())
        df['Gain'] = diff.where(diff > 0, 0)
        df['Loss'] = abs(diff.where(diff < 0, 0))
        df = RSI(df)


        runs = 30
        Value = np.zeros(runs)
        Best_genes = np.zeros([runs, 6])
        for j in range(0, runs):
            print("\n\n--------------Run ", j, "-------------------------")
            Best_genes[j], Value[j] = EA(toolbox)
        MAX = np.max(Value)
        indmax = np.argmax(Value)
        MIN  = np.min(Value)
        STD = np.std(Value)
        Mean = np.mean(Value)
        print("MAX:", MAX)
        print("MIN:", MIN)
        print("Mean:", Mean)
        print("STD:", STD)
        print("Best genes:", Best_genes[indmax])
        np.save('Resultados/Best_genes/Best_' + i +'.npy', Best_genes)
        np.save('Resultados/Value/Val_'+ i +'.npy', Value)
        plt.figure(figsize=(12,10))
        plt.boxplot(Value)
    plt.show()

   