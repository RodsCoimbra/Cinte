import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools
from sys import exit
import random
import bisect

def ROI_total(Roi_short_sum, Roi_long_sum):
    return (Roi_short_sum + Roi_long_sum) / 2

def ROI_long(sell, buy):
    return ((sell - buy) / buy) * 100


def ROI_short(sell, buy):
    return ((sell - buy) / sell) * 100


def drawdown(max, min):
    return abs((max - min) / max) *100

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

# Aceder aos dados, retornando o valor do index em que acontece a tranposição do limite [(Period)*21+(LImite)+63*(long:1, short:0)]
def dados_index(RSI, limit, type, index):
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

#flag= True se o primeiro é min e False se o primeiro é minimo
def drawdown_flag_init(idx):
    while True: #Descobre se o primeiro é maximo ou minimo, caso seja igual continua até encontrar um diferente
        if(idx+1 >= df['Close'].size): #Caso em que o primeiro é o ultimo valor do array
            return False
        
        if(df_close_value(idx) > df_close_value(idx + 1)): #Caso em que o primeiro é máximo
            flag = False        #Quando False quer dizer que o próximo a encontrar é um minimo
            break

        elif(df_close_value(idx) < df_close_value(idx + 1)): #Caso em que o primeiro é minimo
            flag = True         #Quando True quer dizer que o próximo a encontrar é um máximo
            break

        idx+=1
    return flag

def find_next_index(Array,idx):
    next_index = bisect.bisect_right(Array, idx)
    return next_index

def df_close_value(idx):
    return df['Close'].iloc[int(idx)]

def Is_out(idx, Array, fim):
    if(idx >= len(Array) or Array[idx] > fim):
        return True
    return False

def drawdown_short(inicio, fim):
    max_drawdown = 0
    flag_max = drawdown_flag_init(inicio)
    inicio_min = False
    if(flag_max == True):                   #Caso em que o primeiro é minimo
        idx_max = find_next_index(Array_max, inicio)

        if(Is_out(idx_max, Array_max, fim)):
            return drawdown(df_close_value(inicio), df_close_value(fim))        #Tudo foi drawdown, porque foi sempre a subir
        max_drawdown = drawdown(df_close_value(inicio), df_close_value(Array_max[idx_max]))
        
        idx_min = find_next_index(Array_min, Array_max[idx_max])
        
        if(Is_out(idx_min, Array_min, fim)):              #Só tem um min
            return  max_drawdown
        
        if(df_close_value(inicio) < df_close_value(Array_min[idx_min])):    #caso o do inicio seja menor continua a testar para esse valor, 
            idx_min-=1                                                      #senão o valor do inicio para o próximo max já foi guardado no drawdown em cima
            temp = Array_min[idx_min]
            temp_idx = idx_min  
            Array_min[idx_min] = inicio
            inicio_min = True
        if(inicio_min == False): 
            idx_max+=1  #Caso esteja a False quer dizer que passamos ao próximo par min-max
    else:
        idx_min = find_next_index(Array_min, inicio)

        if(Is_out(idx_min, Array_min, fim)):
            return 0                        #Não há drawdown, porque foi sempre a descer
        
        idx_max = find_next_index(Array_max, Array_min[idx_min])

    idx_min_aux = idx_min + 1
    idx_max_aux = idx_max +1
    DD = 0
    while True:
            if(Is_out(idx_min, Array_min, fim)):          #Condições de saida quando termina em minimo local
                break

            elif(Is_out(idx_max, Array_max, fim)):    #Condições de saida quando termina em máximo local
                DD = drawdown(df_close_value(Array_min[idx_min]), df_close_value(fim))
                break
            
            elif(Is_out(idx_min_aux, Array_min, fim)):  #Condições para garantir que se encontra dentro do array o próximo min
                DD = drawdown(df_close_value(Array_min[idx_min]), df_close_value(Array_max[idx_max])) #Caso não esteja faz um ultimo drawdown com o min atual  
                break

            else:
                if(df_close_value(Array_min[idx_min]) < df_close_value(Array_min[idx_min_aux])): #Caso em que o próximo minimo é menor que o atual
                    
                    if(not(Is_out(idx_max_aux, Array_max, fim)) and df_close_value(Array_max[idx_max]) < df_close_value(Array_max[idx_max_aux])):  #Próximo máximo é maior que o atual
                        DD = drawdown(df_close_value(Array_min[idx_min]), df_close_value(Array_max[idx_max_aux]))

                    else:                       #Caso em que o minimo atual é menor que o próximo, então faz o drawdown com o atual
                        DD = drawdown(df_close_value(Array_min[idx_min]), df_close_value(Array_max[idx_max]))
                
                else:                           #Caso em que já temos o maior drawdown até ao próximo máximo
                    DD = drawdown(df_close_value(Array_min[idx_min]), df_close_value(Array_max[idx_max]))
                    idx_min = idx_min_aux

                if(DD > max_drawdown):
                    max_drawdown = DD
                idx_max += 1
                idx_min_aux +=1
                idx_max_aux +=1
    #Saiu do while           
    if(DD > max_drawdown):
        max_drawdown = DD
    if(inicio_min):             #Restaura o valor ao Array_max caso tenha sido alterado
        Array_min[temp_idx] = temp
    return max_drawdown 


def drawdown_long(inicio, fim):
    max_drawdown = 0
    flag_max = drawdown_flag_init(inicio)
    inicio_max = False                      #Quando verdadeiro então o valor do inicio é maior que o próximo máximo
    if(flag_max == True):                   #Caso em que o primeiro é minimo
        idx_max = find_next_index(Array_max, inicio)

        if(Is_out(idx_max, Array_max, fim)):
            return 0                        #Não há drawdown, porque foi sempre a subir
        idx_min = find_next_index(Array_min, Array_max[idx_max])

    else:                   #Caso em que o primeiro é máximo   
        idx_min = find_next_index(Array_min, inicio)

        if(Is_out(idx_min, Array_min, fim)):
            return drawdown(df_close_value(inicio), df_close_value(fim))        #Tudo foi drawdown, porque foi sempre a descer
        max_drawdown = drawdown(df_close_value(inicio), df_close_value(Array_min[idx_min]))
        
        idx_max = find_next_index(Array_max, Array_min[idx_min])
        
        if(Is_out(idx_max, Array_max, fim)):              #Só tem um min
            return  max_drawdown
        
        if(df_close_value(inicio) > df_close_value(Array_max[idx_max])):    #caso o do inicio seja maior continua a testar para esse valor, 
            idx_max-=1                                                      #senão o valor do inicio para o próximo min já foi guardado no drawdown em cima
            temp = Array_max[idx_max]
            temp_idx = idx_max  
            Array_max[idx_max] = inicio
            inicio_max = True
        if(inicio_max == False): 
            idx_min+=1  #Caso esteja a False quer dizer que passamos ao próximo par max-min

    idx_min_aux = idx_min + 1
    idx_max_aux = idx_max +1
    DD = 0
    while True:
            if(Is_out(idx_max, Array_max, fim)):     #Condições de saida quando termina em maximo local
                break

            elif(Is_out(idx_min, Array_min, fim)):                                          #Condições de saida quando termina em minimo local
                DD = drawdown(df_close_value(Array_max[idx_max]), df_close_value(fim))
                break

            elif(Is_out(idx_max_aux, Array_max, fim)):  #Condições para garantir que se encontra dentro do array o próximo max
                DD = drawdown(df_close_value(Array_max[idx_max]), df_close_value(Array_min[idx_min]))   #Caso não esteja faz um ultimo drawdown com o max atual  
                break

            else:              
                if(df_close_value(Array_max[idx_max]) > df_close_value(Array_max[idx_max_aux])): #Caso em que o próximo máximo é menor que o atual

                    if(not(Is_out(idx_min_aux, Array_min, fim)) and df_close_value(Array_min[idx_min]) > df_close_value(Array_min[idx_min_aux])):  #Próximo minimo é menor que o atual
                        DD = drawdown(df_close_value(Array_max[idx_max]), df_close_value(Array_min[idx_min_aux]))

                    else:             #Caso em que o minimo atual é menor que o próximo ou o próximo está fora do intervalo, então faz o drawdown com o atual
                        DD = drawdown(df_close_value(Array_max[idx_max]), df_close_value(Array_min[idx_min]))
                
                else:                           #Caso em que já temos o maior drawdown até ao próximo máximo
                    DD = drawdown(df_close_value(Array_max[idx_max]), df_close_value(Array_min[idx_min]))
                    idx_max = idx_max_aux

                if(DD > max_drawdown):
                    max_drawdown = DD
                idx_min += 1
                idx_min_aux +=1
                idx_max_aux +=1

    #Saiu do while           
    if(DD > max_drawdown):
        max_drawdown = DD
    if(inicio_max):             #Restaura o valor ao Array_max caso tenha sido alterado
        Array_max[temp_idx] = temp
    return max_drawdown                
        
        
def short_results(RSI_short, lls, uls, type):
    flag = False
    Roi_short = []
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
                if (venda_final == True or dados_index(RSI_short, lls, 'short', index_buy) < dados_index(RSI_short, uls, 'short', index_sell)):  
                    if(index_buy+1 >= dados_size(RSI_short, lls, 'short')):
                        sell_short = df_close_value(dados_index(RSI_short, uls, 'short', index_sell))
                        buy_short = df_close_value(df[RSI_period_short].size-1)
                        Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))   
                        break         

                    else:                 #Caso em que não terminou o array do buy
                        index_buy+=1  

                else:
                    if(dados_index(RSI_short, lls, 'short', index_buy) == dados_index(RSI_short, uls, 'short', index_sell)):
                        index_sell+=1
                        continue
                    sell_short = df_close_value(dados_index(RSI_short, uls, 'short', index_sell))
                    flag = True
                    
        else:               
            buy_short = df_close_value(dados_index(RSI_short, lls, 'short', index_buy))
            flag = False
            Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))
            while (dados_index(RSI_short, lls, 'short', index_buy) > dados_index(RSI_short, uls, 'short', index_sell)):
                        if (index_sell+1 >= dados_size(RSI_short, uls, 'short')):
                            index_sell+=1
                            break
                        else:
                            index_sell+=1
    
    return np.sum(Roi_short)

def long_results(RSI_long, lll, ull, type):
    RSI_period_long = 'RSI_' + str((RSI_long+1)*7)
    index_buy = 0
    index_sell = 0  
    Roi_long = []
    flag_sell = False           #Flag de venda
    venda_final = False     
    if(dados_size(RSI_long, ull, 'long') == 0):
        venda_final = True      #Casos em que nunca passa o limite superior do long
    while True:
        if(flag_sell == False):      #Caso em que estamos a comprar
            if (index_buy >= dados_size(RSI_long, lll, 'long')):        #Caso em que já não há mais para comprar
                break
            
            else:
                if (venda_final == True or dados_index(RSI_long, lll, 'long', index_buy) > dados_index(RSI_long, ull, 'long', index_sell)):     #Tem uma venda antes de comprar
                    if(index_sell+1 >= dados_size(RSI_long, ull, 'long')):                #Caso já não haja próximo sell, logo vende no indice final                                                
                        buy_long = df_close_value(dados_index(RSI_long, lll, 'long', index_buy))
                        sell_long = df_close_value(df[RSI_period_long].size-1)
                        Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
                        break 
                    else:                            #Passa para o index seguinte do sell   
                        index_sell+=1   

                elif(dados_index(RSI_long, lll, 'long', index_buy) == dados_index(RSI_long, ull, 'long', index_sell)):  #Caso em que o buy é igual ao sell, logo não realiza transação
                    index_buy+=1
                    continue
                else:
                    buy_long = df_close_value(dados_index(RSI_long, lll, 'long', index_buy))            #Compra
                    flag_sell = True

        else:                   #Vende
            sell_long = df_close_value(dados_index(RSI_long, ull, 'long', index_sell))
            flag_sell = False
            Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
            while (dados_index(RSI_long, lll, 'long', index_buy) < dados_index(RSI_long, ull, 'long', index_sell)): #Procura o próximo endereço de compra possivel
                        if (index_buy+1 >= dados_size(RSI_long, lll, 'long')):
                            index_buy+=1
                            break
                        else:
                            index_buy+=1

    return np.sum(Roi_long)


def ROI_results(RSI_short, RSI_long, lll, ull, lls, uls, type):
    short = short_results(RSI_short, lls, uls, type)
    #Para apagar depois de mexer no drawdown
    # short2 = short_results2(RSI_short, lls, uls)
    # if(short != short2):
    #     print("ERRO no short")
    #     exit(1)
    long = long_results(RSI_long, lll, ull, type)
    # long2 = long_results2(RSI_long, lll, ull)
    # if(long != long2):
    #     print("ERRO no long")
    #     exit(1)
    return ROI_total(short, long)


def pre_processing_drawdown(df):
    Array_max = []
    Array_min = []
    flag_max = drawdown_flag_init(0)
    if (flag_max == True):
        Array_min = np.append(Array_min, 0)
    else:
        Array_max = np.append(Array_max, 0)

    for idx, value in enumerate(df['Close']):
        if(df['Close'].size == idx+1):
            break               
        if(flag_max == True and value > df_close_value(idx+1)):
            Array_max = np.append(Array_max, idx) 
            flag_max = False      
        elif(flag_max == False and value < df_close_value(idx+1)):
            Array_min = np.append(Array_min, idx)
            flag_max = True
    return Array_max, Array_min

# Ordenação do array dados:[(Period)*21+(LImite)+63*(long:1, short:0)] (Acedido por função dados_index)
def pre_processing_dados(df):
    dados = [[] for _ in range(21*3*2)]
    index = -1
    #Dados de todos os indices onde passa os limites do long e do short
    for i in range(7,22,7):
        RSI_period= 'RSI_' + str(i)
        for j in range(0,101,5):
            index+=1
            for idx, value in enumerate(df[RSI_period]): #0-62 short
                if(df[RSI_period].size == idx+1):
                    break
                else:
                    if (value > j and df[RSI_period].iloc[idx+1] <= j):
                        dados[index] = np.append(dados[index],idx+1)
                    if (value < j and df[RSI_period].iloc[idx+1] >= j): #63-125 long
                        dados[index+63] = np.append(dados[index+63],idx+1)  # mais 63 porque os primeiros 62 indices (21_limits*3_rsi_period) são do short
    return dados

def evaluate(individual):
    genes = np.zeros(6, dtype=int)
    for idx, vars in enumerate(individual):
        genes[idx] = vars
    return ROI_results(genes[0], genes[1], genes[2], genes[3], genes[4], genes[5], 0),
    
def create_EA():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("RSI_short", random.randint, 0, 2)
    toolbox.register("RSI_long", random.randint, 0, 2)
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
    # g_early = 0
    while g < 100:
        # A new generation 
        g = g + 1
        
        
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
                if (mutant[0]>2):
                    mutant[0]%= 3
                if (mutant[1]>2):
                    mutant[1]%=3 
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
        # length = len(pop)
        # mean = sum(fits) / length
        # sum2 = sum(x*x for x in fits)
        # std = abs(sum2 / length - mean**2)**0.5
        
        if(g%20==0):
            print("-- Generation %i --" % g)
            print("  Max %s" % Max_early)
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)

        if (Max_early < max(fits)):
            Max_early = max(fits)
            best_ind = tools.selBest(pop, 1)[0]
            # g_early = g
        # if (g - g_early > 40):
        #     print("\n-------------------------------Early stop---------------------------")
        #     break        
    best_genes = np.array(best_ind)
    print("Best individual is",(best_genes[0:2]+1) * 7,best_genes[2:]*5, best_ind.fitness.values[0])
    return np.append ((best_genes[0:2]+1) * 7, best_genes[2:]*5), best_ind.fitness.values[0]


# def short_results2(RSI_short, lls, uls):
#     flag = False
#     Roi_short = []
#     index_buy = 0
#     index_sell = 0  
#     RSI_period_short = 'RSI_' + str((RSI_short+1)*7)
#     venda_final = False
#     if(dados_size(RSI_short, lls, 'short') == 0):
#         venda_final = True
#     while True:
#         if(flag == False):
#             if (index_sell >= dados_size(RSI_short, uls, 'short')):
#                 break
#             else:
#                 if (venda_final == True or dados_index(RSI_short, lls, 'short', index_buy) < dados_index(RSI_short, uls, 'short', index_sell)):  
#                     if(index_buy+1 >= dados_size(RSI_short, lls, 'short')):
#                         sell_short = df_close_value(dados_index(RSI_short, uls, 'short', index_sell))
#                         buy_short = df_close_value(df[RSI_period_short].size-1)
#                         Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))   
#                         break         

#                     else:                 #Caso em que não terminou o array do buy
#                         index_buy+=1  

#                 else:
#                     if(dados_index(RSI_short, lls, 'short', index_buy) == dados_index(RSI_short, uls, 'short', index_sell)):
#                         index_sell+=1
#                         continue
#                     sell_short = df_close_value(dados_index(RSI_short, uls, 'short', index_sell))
#                     flag = True
#                     while (dados_index(RSI_short, lls, 'short', index_buy) > dados_index(RSI_short, uls, 'short', index_sell)):
#                         if (index_sell+1 >= dados_size(RSI_short, uls, 'short')):
#                             index_sell+=1
#                             break
#                         else:
#                             index_sell+=1
#         else:               
#             buy_short = df_close_value(dados_index(RSI_short, lls, 'short', index_buy))
#             flag = False
#             Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))
    
#     return np.sum(Roi_short)

# def long_results2(RSI_long, lll, ull):
#     RSI_period_long = 'RSI_' + str((RSI_long+1)*7)
#     index_buy = 0
#     index_sell = 0  
#     Roi_long = []
#     flag = False
#     venda_final = False
#     if(dados_size(RSI_long, ull, 'long') == 0):
#         venda_final = True
#     while True:
#         if(flag == False):
#             if (index_buy >= dados_size(RSI_long, lll, 'long')):
#                 break
#             else:
#                 if (venda_final == True or dados_index(RSI_long, lll, 'long', index_buy) > dados_index(RSI_long, ull, 'long', index_sell)):  
#                     if(index_sell+1 >= dados_size(RSI_long, ull, 'long')):                #Se tiver mais algum para vender então vende                                                  
#                         buy_long = df_close_value(dados_index(RSI_long, lll, 'long', index_buy))
#                         sell_long = df_close_value(df[RSI_period_long].size-1)
#                         Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
#                         break 
#                     else:                           #Caso em que não terminou o array do sell        
#                         index_sell+=1
#                 else:
#                     if(dados_index(RSI_long, lll, 'long', index_buy) == dados_index(RSI_long, ull, 'long', index_sell)):
#                         index_buy+=1
#                         continue
#                     buy_long = df_close_value(dados_index(RSI_long, lll, 'long', index_buy))
#                     flag = True
#                     while (dados_index(RSI_long, lll, 'long', index_buy) < dados_index(RSI_long, ull, 'long', index_sell)):
#                         if (index_buy+1 >= dados_size(RSI_long, lll, 'long')):
#                             index_buy+=1
#                             break
#                         else:
#                             index_buy+=1
#         else:
#             sell_long = df_close_value(dados_index(RSI_long, ull, 'long', index_sell))
#             flag = False
#             Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))

#     return np.sum(Roi_long)



if __name__ == '__main__':
    # path = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F',
    #          'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']
    df = {}
    path = ['AAPL']
    # path = ['Teste']
    runs = 5
    toolbox = create_EA()
    # Read data from csv files
    # start = pd.to_datetime('01-01-2011', dayfirst=True)
    # end = pd.to_datetime('31-12-2019', dayfirst=True)
    start = pd.to_datetime('01-01-2020', dayfirst=True)
    end = pd.to_datetime('31-12-2022', dayfirst=True)
    # start = pd.to_datetime('01-08-2023', dayfirst=True)
    # end = pd.to_datetime('15-09-2023', dayfirst=True)
    # start = pd.to_datetime('01-01-2023', dayfirst=True)
    # end = pd.to_datetime('01-06-2023', dayfirst=True) 
        
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
        dados = pre_processing_dados(df)
        Array_max, Array_min = pre_processing_drawdown(df)
        Value = np.zeros(runs)
        Best_genes = np.zeros([runs, 6]) 


        # plt.plot(df['Date'].iloc[51:68], df['Close'].iloc[51:68])
        # plt.plot(df['Date'], df['Close'])  
        # plt.scatter(df['Date'].iloc[Array_max.astype(int)], df['Close'].iloc[Array_max.astype(int)], color='red')
        # plt.scatter(df['Date'].iloc[Array_min.astype(int)], df['Close'].iloc[Array_min.astype(int)], color='green')
        # plt.show()   


        random.seed(100)
        # for i in range(0,1000):
        #     a = random.randint(0, df['Close'].size-1)
        #     b = random.randint(0, df['Close'].size-1)
        #     if(a > b):
        #         a,b = b,a
        #     if(find_max_drawdown(df['Close'], a, b) != drawdown_long(a, b)):
        #         print(i, "-> ",a,b,  find_max_drawdown(df['Close'], a, b), drawdown_long(a, b))  


        
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
        # np.save('Resultados/Best_genes/Besttrain_' + i +'.npy', Best_genes)
        # np.save('Resultados/Value/Valtrain_'+ i +'.npy', Value)
        # plt.figure(figsize=(12,10))
        # plt.boxplot(Value)
    # plt.show()