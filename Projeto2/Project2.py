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

#Max = max se long e Max = min se short
def drawdown(max, min):
    return ((max - min) / max) *100

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


def dados_value(RSI, limit, type, index):
    if(type == 'short' or type == 0):
        return int(dados[RSI*21+limit][index])
    if(type == 'long' or type == 1):
        return int(dados[RSI*21+limit+63][index])
    else:
        print("ERRO, type mal")
        exit(1)

def dados_size(RSI, limit, type):
    if(type == 'short' or type == 0):
        if len(dados[RSI*21+limit]) == 0:
            return 0
        return dados[RSI*21+limit].size
    if(type == 'long' or type == 1):
        if len(dados[RSI*21+limit+63]) == 0:
            return 0
        return dados[RSI*21+limit+63].size
    else:
        print("ERRO, type mal")
        exit(1)

def ROI_results(RSI_short, RSI_long, lll, ull, lls, uls):

    #short
    #print(7*(1+RSI_short), 7*(1+RSI_long), 5*lll, 5*ull, 5*lls, 5*uls, "\n")
    #print(RSI_short, RSI_long, lll, ull, lls, uls, "\n")
    flag = False
    Roi_short = []
    sell_short = 0
    index_buy = 0
    index_sell = 0  
    RSI_period_short = 'RSI_' + str((RSI_short+1)*7)
    venda_final = False
    if(dados_size(RSI_short, lls, 'short') == 0):
        venda_final = True
    while True:
        if(flag == False):
            if (index_sell >= dados_size(RSI_short, uls, 'short')):
                break
            else:
                if (venda_final == True or dados_value(RSI_short, lls, 'short', index_buy) < dados_value(RSI_short, uls, 'short', index_sell)):  
                    if(index_buy+1 >= dados_size(RSI_short, lls, 'short')):
                        sell_short = df['Close'].iloc[dados_value(RSI_short, uls, 'short', index_sell)]
                        buy_short = df['Close'].iloc[df[RSI_period_short].size-1]
                        Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))   
                        break         

                    else:                 #Caso em que não terminou o array do buy
                        index_buy+=1  

                else:
                    if(dados_value(RSI_short, lls, 'short', index_buy) == dados_value(RSI_short, uls, 'short', index_sell)):
                        index_sell+=1
                        continue
                    sell_short = df['Close'].iloc[dados_value(RSI_short, uls, 'short', index_sell)]
                    flag = True
                    while (dados_value(RSI_short, lls, 'short', index_buy) > dados_value(RSI_short, uls, 'short', index_sell)):
                        if (index_sell+1 >= dados_size(RSI_short, uls, 'short')):
                            index_sell+=1
                            break
                        else:
                            index_sell+=1
        else:               
            buy_short = df['Close'].iloc[dados_value(RSI_short, lls, 'short', index_buy)]
            flag = False
            Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))

    #long
    RSI_period_long = 'RSI_' + str((RSI_long+1)*7)
    index_buy = 0
    index_sell = 0  
    Roi_long = []
    flag = False
    venda_final = False
    if(dados_size(RSI_long, ull, 'long') == 0):
        venda_final = True
    while True:
        if(flag == False):
            if (index_buy >= dados_size(RSI_long, lll, 'long')):
                break
            else:
                if (venda_final == True or dados_value(RSI_long, lll, 'long', index_buy) > dados_value(RSI_long, ull, 'long', index_sell)):  
                    if(index_sell+1 >= dados_size(RSI_long, ull, 'long')):                #Se tiver mais algum para vender então vende                                                  
                        buy_long = df['Close'].iloc[dados_value(RSI_long, lll, 'long', index_buy)]
                        sell_long = df['Close'].iloc[df[RSI_period_long].size-1]
                        Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
                        break 
                    else:                           #Caso em que não terminou o array do sell        
                        index_sell+=1
                else:
                    if(dados_value(RSI_long, lll, 'long', index_buy) == dados_value(RSI_long, ull, 'long', index_sell)):
                        index_buy+=1
                        continue
                    buy_long = df['Close'].iloc[dados_value(RSI_long, lll, 'long', index_buy)]
                    flag = True
                    while (dados_value(RSI_long, lll, 'long', index_buy) < dados_value(RSI_long, ull, 'long', index_sell)):
                        if (index_buy+1 >= dados_size(RSI_long, lll, 'long')):
                            index_buy+=1
                            break
                        else:
                            index_buy+=1
        else:
            sell_long = df['Close'].iloc[dados_value(RSI_long, ull, 'long', index_sell)]
            flag = False
            Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))

    # print("ROI_short:", np.sum(Roi_short))
    # print("ROI_long:", np.sum(Roi_long))
    return ROI_total(np.sum(Roi_short), np.sum(Roi_long))


# Aceder aos dados[(Period)*21+(LImite)+63*(long:1, short:0)]
def pre_processing(df, dados):
    index = -1
    for i in range(7,22,7):
        RSI_period= 'RSI_' + str(i)
        for j in range(0,101,5):
            index+=1
            for idx, value in enumerate(df[RSI_period]): #0-62 short
                if(df[RSI_period].size == idx+1):
                    break
                else:
                    if (value > j and df[RSI_period].iloc[idx+1] < j):
                        dados[index] = np.append(dados[index],idx+1)
                    if (value < j and df[RSI_period].iloc[idx+1] > j): #63-125 long
                        dados[index+63] = np.append(dados[index+63],idx+1)  # mais 63 porque os primeiros 62 indices (21_limits*3_rsi_period) são do short



def evaluate(individual):
    genes = np.zeros(6, dtype=int)
    for idx, vars in enumerate(individual):
        genes[idx] = vars
    return ROI_results(genes[0]-1, genes[1]-1, genes[2], genes[3], genes[4], genes[5]),
    

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
    Max_early = -1000
    g_early = 0
    while g < 100:
        # A new generation 
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
            print("  Max %s" % Max_early)
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        if (Max_early < max(fits)):
            Max_early = max(fits)
            best_ind = tools.selBest(pop, 1)[0]
            # g_early = g
        # if (g - g_early > 40):
        #     print("\n-------------------------------Early stop---------------------------")
        #     break
    best_genes = np.array(best_ind)
    print("Best individual is",best_genes[0:2] * 7,best_genes[2:]*5, best_ind.fitness.values[0])
    return np.append (best_genes[0:2] * 7, best_genes[2:]*5), best_ind.fitness.values[0]

def drawdown_long(df, index1, index2):
    min = float('inf')
    flag_min = False
    for i in range(index1, index2):
        if(flag_min == False and df['Close'].iloc[i] > df['Close'].iloc[i+1]):
            max_local = df['Close'].iloc[i]
            flag_min = True
        elif(flag_min == True and df['Close'].iloc[i] < df['Close'].iloc[i+1]):
            min_local = df['Close'].iloc[i]
            flag_min = False
            drawdown_long = drawdown(max_local, min_local)

def ROI_Dd_results(RSI_short, RSI_long, lll, ull, lls, uls):
    
    flag = False
    Roi_short = []
    sell_short = 0
    index_buy = 0
    index_sell = 0  
    RSI_period_short = 'RSI_' + str((RSI_short+1)*7)
    venda_final = False
    if(dados_size(RSI_short, lls, 'short') == 0):
        venda_final = True
    while True:
        if(flag == False):
            if (index_sell >= dados_size(RSI_short, uls, 'short')):
                break
            else:
                if (venda_final == True or dados_value(RSI_short, lls, 'short', index_buy) < dados_value(RSI_short, uls, 'short', index_sell)):  
                    if(index_buy+1 >= dados_size(RSI_short, lls, 'short')):
                        sell_short = df['Close'].iloc[dados_value(RSI_short, uls, 'short', index_sell)]
                        buy_short = df['Close'].iloc[df[RSI_period_short].size-1]
                        Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))   
                        break         

                    else:                 #Caso em que não terminou o array do buy
                        index_buy+=1  

                else:
                    if(dados_value(RSI_short, lls, 'short', index_buy) == dados_value(RSI_short, uls, 'short', index_sell)):
                        index_sell+=1
                        continue
                    sell_short = df['Close'].iloc[dados_value(RSI_short, uls, 'short', index_sell)]
                    flag = True
                    while (dados_value(RSI_short, lls, 'short', index_buy) > dados_value(RSI_short, uls, 'short', index_sell)):
                        if (index_sell+1 >= dados_size(RSI_short, uls, 'short')):
                            index_sell+=1
                            break
                        else:
                            index_sell+=1
        else:               
            buy_short = df['Close'].iloc[dados_value(RSI_short, lls, 'short', index_buy)]
            flag = False
            Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))

    #long
    RSI_period_long = 'RSI_' + str((RSI_long+1)*7)
    index_buy = 0
    index_sell = 0  
    Roi_long = []
    flag = False
    venda_final = False
    if(dados_size(RSI_long, ull, 'long') == 0):
        venda_final = True
    while True:
        if(flag == False):
            if (index_buy >= dados_size(RSI_long, lll, 'long')):
                break
            else:
                if (venda_final == True or dados_value(RSI_long, lll, 'long', index_buy) > dados_value(RSI_long, ull, 'long', index_sell)):  
                    if(index_sell+1 >= dados_size(RSI_long, ull, 'long')):                #Se tiver mais algum para vender então vende                                                  
                        buy_long = df['Close'].iloc[dados_value(RSI_long, lll, 'long', index_buy)]
                        sell_long = df['Close'].iloc[df[RSI_period_long].size-1]
                        Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
                        break 
                    else:                           #Caso em que não terminou o array do sell        
                        index_sell+=1
                else:
                    if(dados_value(RSI_long, lll, 'long', index_buy) == dados_value(RSI_long, ull, 'long', index_sell)):
                        index_buy+=1
                        continue
                    buy_long = df['Close'].iloc[dados_value(RSI_long, lll, 'long', index_buy)]
                    flag = True
                    while (dados_value(RSI_long, lll, 'long', index_buy) < dados_value(RSI_long, ull, 'long', index_sell)):
                        if (index_buy+1 >= dados_size(RSI_long, lll, 'long')):
                            index_buy+=1
                            break
                        else:
                            index_buy+=1
        else:
            sell_long = df['Close'].iloc[dados_value(RSI_long, ull, 'long', index_sell)]
            flag = False
            Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
    return ROI_total(np.sum(Roi_short), np.sum(Roi_long))

if __name__ == '__main__':
    path = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F',
             'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']
    df = {}

    toolbox = create_EA()
    # Read data from csv files
    # start = pd.to_datetime('01-01-2020', dayfirst=True)
    # end = pd.to_datetime('31-12-2022', dayfirst=True)
    start = pd.to_datetime('01-08-2023', dayfirst=True)
    end = pd.to_datetime('15-09-2023', dayfirst=True)
     
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
        dados = [[] for _ in range(21*3*2)]   
        pre_processing(df,dados)
        df.to_csv("df.csv")
        runs = 30
        Value = np.zeros(runs)
        Best_genes = np.zeros([runs, 6])





    #     for j in range(0, runs):
    #         print("\n\n--------------Run ", j, "-------------------------")
    #         Best_genes[j], Value[j] = EA(toolbox)
    #     MAX = np.max(Value)
    #     indmax = np.argmax(Value)
    #     MIN  = np.min(Value)
    #     STD = np.std(Value)
    #     Mean = np.mean(Value)
    #     print("MAX:", MAX)
    #     print("MIN:", MIN)
    #     print("Mean:", Mean)
    #     print("STD:", STD)
    #     print("Best genes:", Best_genes[indmax])
    #     np.save('Resultados/Best_genes/Best_' + i +'.npy', Best_genes)
    #     np.save('Resultados/Value/Val_'+ i +'.npy', Value)
    #     plt.figure(figsize=(12,10))
    #     plt.boxplot(Value)
    # plt.show()