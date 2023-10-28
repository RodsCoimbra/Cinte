from pickle import TRUE
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

def RSI(df):        #Calcula o RSI para 7, 14 e 21 dias e mete no dataframe
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
    drawdown_max = 0
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
                        sell_short_aux = dados_index(RSI_short, uls, 'short', index_sell)
                        sell_short = df_close_value(sell_short_aux)
                        buy_short = df_close_value(df[RSI_period_short].size-1)
                        if(type == 1):
                            DD = drawdown_short(sell_short_aux, df[RSI_period_short].size-1)
                            if(DD > drawdown_max):
                                drawdown_max = DD
                        Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))   
                        break         

                    else:                 #Caso em que não terminou o array do buy
                        index_buy+=1  

                else:
                    if(dados_index(RSI_short, lls, 'short', index_buy) == dados_index(RSI_short, uls, 'short', index_sell)):
                        index_sell+=1
                        continue
                    sell_short_aux = dados_index(RSI_short, uls, 'short', index_sell)
                    sell_short = df_close_value(sell_short_aux)
                    flag = True
                    
        else:     
            buy_short_aux = dados_index(RSI_short, lls, 'short', index_buy)
            buy_short = df_close_value(buy_short_aux)
            flag = False
            Roi_short = np.append(Roi_short, ROI_short(sell_short, buy_short))
            if(type == 1):
                DD = drawdown_short(sell_short_aux, buy_short_aux)
                if(DD > drawdown_max):
                    drawdown_max = DD
            while (dados_index(RSI_short, lls, 'short', index_buy) > dados_index(RSI_short, uls, 'short', index_sell)):
                        if (index_sell+1 >= dados_size(RSI_short, uls, 'short')):
                            index_sell+=1
                            break
                        else:
                            index_sell+=1
    return np.sum(Roi_short), drawdown_max

def long_results(RSI_long, lll, ull, type):
    drawdown_max = 0
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
                        index_buy_aux = dados_index(RSI_long, lll, 'long', index_buy)
                        buy_long = df_close_value(index_buy_aux)
                        sell_long = df_close_value(df[RSI_period_long].size-1)
                        if(type == 1):
                            DD = drawdown_long(index_buy_aux, df[RSI_period_long].size-1)
                            if(DD > drawdown_max):
                                drawdown_max = DD
                        Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
                        break 
                    else:                            #Passa para o index seguinte do sell   
                        index_sell+=1   

                elif(dados_index(RSI_long, lll, 'long', index_buy) == dados_index(RSI_long, ull, 'long', index_sell)):  #Caso em que o buy é igual ao sell, logo não realiza transação
                    index_buy+=1
                    continue
                else:
                    index_buy_aux = dados_index(RSI_long, lll, 'long', index_buy)
                    buy_long = df_close_value(index_buy_aux)            #Compra
                    flag_sell = True

        else:                   #Vende
            index_sell_aux = dados_index(RSI_long, ull, 'long', index_sell)
            sell_long = df_close_value(index_sell_aux)
            flag_sell = False
            Roi_long = np.append(Roi_long, ROI_long(sell_long, buy_long))
            if(type == 1):
                DD = drawdown_long(index_buy_aux, index_sell_aux)
                if(DD > drawdown_max):
                    drawdown_max = DD
            while (dados_index(RSI_long, lll, 'long', index_buy) < dados_index(RSI_long, ull, 'long', index_sell)): #Procura o próximo endereço de compra possivel
                        if (index_buy+1 >= dados_size(RSI_long, lll, 'long')):
                            index_buy+=1
                            break
                        else:
                            index_buy+=1

    return np.sum(Roi_long), drawdown_max


def ROI_results(RSI_short, RSI_long, lll, ull, lls, uls, type):

    short, DD_short = short_results(RSI_short, lls, uls, type)
    long, DD_long = long_results(RSI_long, lll, ull, type)
    
    if(type == 0):
        return ROI_total(short, long)
    if(DD_long > DD_short):
        return ROI_total(short, long), DD_long
    else:
        return ROI_total(short, long), DD_short


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
    

def evaluate_nsga2(individual):
    genes = np.zeros(6, dtype=int)
    for idx, vars in enumerate(individual):
        genes[idx] = vars
    ROI, max_drawdown = ROI_results(genes[0], genes[1], genes[2], genes[3], genes[4], genes[5], 1)
    return ROI, max_drawdown 


def create_EA(nsga2):
    
    if nsga2:
        # Minimize (-1) or maximize (1) ROI and drawdown
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    else:
        # Maximize ROI
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
    toolbox.register("evaluate_nsga2", evaluate_nsga2)

    toolbox.register("mate", tools.cxTwoPoint)    #Two point crossover

    toolbox.register("mutate", tools.mutUniformInt, low= 0, up= 20, indpb=0.2) 

    if nsga2:
        toolbox.register('select', tools.selNSGA2)
    else:
        toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def EA(toolbox, nsga2):
    HallofFame = []
    CXPB, MUTPB, pop_size, runs = 0.6, 0.35, 100, 100
    if nsga2:
        pop_size, runs = 64, 150  #Caso seja nsga2 a população tem de ser 64(pedido no enunciado) e portanto aumentamos as runs

    pop = toolbox.population(n=pop_size)

    if nsga2:
        HallofFame = tools.ParetoFront()               
        fitnesses = list(map(toolbox.evaluate_nsga2, pop))
        
    else:    
        fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    if nsga2:
        pop = toolbox.select(pop, len(pop))
        
    g = 0
    Max_early = -1000
    # g_early = 0
    while g < runs:
        # A new generation 
        g = g + 1
        
        if nsga2:
            offspring = tools.selTournamentDCD(pop, len(pop))
        else:
            offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() <= CXPB:
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
        if nsga2:
            fitnesses = list(map(toolbox.evaluate_nsga2, invalid_ind))
        else:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # The population is entirely replaced by the offspring
        if nsga2:
            pop = toolbox.select(pop + offspring, pop_size)
            HallofFame.update(pop) 
        else:
            pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        if nsga2:
            fits2 = [ind.fitness.values[1] for ind in pop]
        
        if not(nsga2):
            if (Max_early < max(fits)):
                Max_early = max(fits)
                best_ind = tools.selBest(pop, 1)[0]
        

    if not(nsga2):  
        best_genes = np.array(best_ind)
        # print("Best individual is",(best_genes[0:2]+1) * 7,best_genes[2:]*5, best_ind.fitness.values[0])
        return best_genes, best_ind.fitness.values[0]
    if nsga2:
        return HallofFame


if __name__ == '__main__':
    #---------------------Variáveis a mudar-------------------------------------------------------------------------
    nsga2 = False      #Mudar para False no caso de querer apenas maximizar o ROI e True para ambos


    #Atenção que o train_test só funciona para maximizar o ROI, logo a flag nsga2 tem de estar a False!
    Train_test = True   #Mudar para False no caso de querer treinar e testar em 2020-2022 e True para treinar em 2011-2019 e testar nos de 2020-2022
    
    # Mudar o path para testar com outros dados
    path = ['AAL', 'AAPL', 'AMZN', 'BAC', 'F', 'GOOG', 'IBM', 'INTC', 'NVDA', 'XOM']
    
    runs = 30  #Número de runs pretendidas

    save_dados = TRUE #Mudar para True no caso de querer guardar os dados para usar em gráficos
                       #Atenção que se True é necessário criar diretórios especificos para não dar erro,
                       #Sendo esses, Resultados/HallofFameNSGA2, Resultados/ValuesNSGA2 e Resultados/Best_genesNSGA2 se nsga2 for True
                       #e Resultados/Best_genes e Resultados/Values se nsga2 for False
    #----------------------------------------------------------------------------------------------------------------


    #30 cores diferentes para o gráfico das pareto fronts
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','lime', 'teal', 'indigo', 'maroon', 'navy', 'salmon', 'gold', 'orchid', 'turquoise', 'slateblue',
        'darkred', 'darkorange', 'forestgreen', 'mediumvioletred', 'saddlebrown', 'darkslategray', 'mediumseagreen', 'mediumslateblue', 'coral', 'cadetblue']
    
    df = {}
    toolbox = create_EA(nsga2)

    if(Train_test):
        start = pd.to_datetime('01-01-2011', dayfirst=True)
        end = pd.to_datetime('31-12-2019', dayfirst=True)
        start2 = pd.to_datetime('01-01-2020', dayfirst=True)
        end2 = pd.to_datetime('31-12-2022', dayfirst=True)
    else:
        start = pd.to_datetime('01-01-2020', dayfirst=True)
        end = pd.to_datetime('31-12-2022', dayfirst=True)
        start2 = None
        end2 = None


    for i in path:      
        print("--------------Path ", i, "-------------------------")
        print("A processar o ficheiro")
        df = pd.read_csv('Data/' + i + '.csv', sep=';', decimal='.', usecols=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df = df[(df['Date'] >= start) & (df['Date'] <= end)] #Limita o intervalo de tempo
        diff = pd.Series(df['Close'].diff())    #Calcula a diferença entre o dia atual e o anterior
        df['Gain'] = diff.where(diff > 0, 0)    #Caso seja positiva é ganho
        df['Loss'] = abs(diff.where(diff < 0, 0))   #Caso seja negativa é perda
        df = RSI(df)
        dados = pre_processing_dados(df)            
        if(Train_test):
            df2 = pd.read_csv('Data/' + i + '.csv', sep=';', decimal='.', usecols=['Date', 'Close'])
            df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)
            df2 = df2[(df2['Date'] >= start2) & (df2['Date'] <= end2)] #Limita o intervalo de tempo
            diff = pd.Series(df2['Close'].diff())    #Calcula a diferença entre o dia atual e o anterior
            df2['Gain'] = diff.where(diff > 0, 0)    #Caso seja positiva é ganho
            df2['Loss'] = abs(diff.where(diff < 0, 0))   #Caso seja negativa é perda
            df2 = RSI(df2)
            dados2 = pre_processing_dados(df2)            
            df_temp = df        #Guarda o valor para depois poder ir trocando o df principal
            dados_temp = dados
        if nsga2:
            plt.figure(str(i))
            Array_max, Array_min = pre_processing_drawdown(df)
            if(Train_test):
                print("Treino e teste não funciona para o NSGA2, como referido nos readme se desejar correr o NSGA2, por favor altere o Train_test para False e volte a correr o programa\n")
                print("Caso contrário se desejar correr o programa para treino e teste, então altere o nsga2 para False e volte a correr o programa\n")
                exit(0)
            if(save_dados):
                Value_Roi = np.zeros([runs, 2])
                Value_DD = np.zeros([runs, 2])
                Best_genes_Roi = np.zeros([runs, 6]) 
                Best_genes_DD =  np.zeros([runs, 6])
                ROI_Hall = []
                DD_Hall = []
                sizes = []

            for j in range(0, runs):
                print("\n\n--------------Run ", j,"  Path ", i,"-------------------------")
                HallofFame = EA(toolbox, nsga2)
                plt.scatter([ind.fitness.values[0] for ind in HallofFame], [ind.fitness.values[1] for ind in HallofFame], color = colors[j]) 
                print("Best ROI:", HallofFame[0].fitness.values[0], HallofFame[0].fitness.values[1])
                print("Best DD:", HallofFame[-1].fitness.values[0],HallofFame[-1].fitness.values[1])
                if(save_dados):
                    Best_genes_Roi[j] = HallofFame[0]
                    Best_genes_DD[j] = HallofFame[-1]
                    Value_Roi[j] = HallofFame[0].fitness.values[0], HallofFame[0].fitness.values[1]
                    Value_DD[j] = HallofFame[-1].fitness.values[0],HallofFame[-1].fitness.values[1]
                    for ind in HallofFame:
                        ROI_Hall = np.append(ROI_Hall, ind.fitness.values[0])
                        DD_Hall = np.append(DD_Hall, ind.fitness.values[1])
                    sizes = np.append(sizes, len(HallofFame))
                
                
            if(save_dados):
                np.save('Resultados/HallofFameNSGA2/size_'+ i +'.npy', sizes)
                np.save('Resultados/HallofFameNSGA2/HallofFame_ROI_'+ i +'.npy', ROI_Hall)
                np.save('Resultados/HallofFameNSGA2/HallofFame_DD_'+ i +'.npy', DD_Hall)
                np.save('Resultados/ValuesNSGA2/Val_ROI_'+ i +'.npy', Value_Roi)
                np.save('Resultados/ValuesNSGA2/Val_DD_'+ i +'.npy', Value_DD)
                np.save('Resultados/Best_genesNSGA2/Best_genes_ROI_'+ i +'.npy', Best_genes_Roi)
                np.save('Resultados/Best_genesNSGA2/Best_genes_DD_'+ i +'.npy', Best_genes_DD)
 
            plt.xlabel('ROI (Maximize)', fontsize=15)
            plt.ylabel('Drawdown (Minimize)', fontsize=15)
            plt.tight_layout()
            plt.show()

            #Caso tenha vários valores maximos de ROI iguais, escolher o melhor Drawdown
            MAX = np.max(Value_Roi[:,0])	
            mask = np.array(Value_Roi[:,0] == MAX)
            masked_array = Value_Roi[mask,1]
            MIN_value_DD = np.min(masked_array) #Menor valor para o maior ROI
                    

            #Caso tenha vários valores minimos de Drawdown iguais, escolher o melhor ROI
            MIN  = np.min(Value_DD[:,1])
            mask2 = np.array(Value_DD[:,1] == MIN)
            masked_array2 = Value_DD[mask2,0]
            Max_value_ROI = np.max(masked_array2)  #Maior valor para o menor drawdown
            print("\n\nValores Finais para o Path ", i, ":")
            print("MAX -> ROI:", MAX, "DD:", MIN_value_DD)
            print("MIN -> ROI:", Max_value_ROI, "DD:", MIN)
        
        else:
            Value = np.zeros(runs)
            Best_genes = np.zeros([runs, 6], dtype=int)
            if(Train_test):
                Value_test = np.zeros(runs)   

            for j in range(0, runs):
                print("\n\n--------------Run ", j,"  Path ", i,"-------------------------")
                Best_genes[j], Value[j] = EA(toolbox, nsga2) 
                if(Train_test):
                    print("Melhor ROI treino:", Value[j])       
                    df = df2
                    dados = dados2
                    Value_test[j] = ROI_results(Best_genes[j][0], Best_genes[j][1], Best_genes[j][2], Best_genes[j][3], Best_genes[j][4], Best_genes[j][5], 0)
                    print("\nROI de teste para a melhor estratégia:", Value_test[j])
                    df = df_temp
                    dados = dados_temp
                else:
                    print("Melhor ROI:", Value[j])


            MAX = np.max(Value)
            MIN  = np.min(Value)
            STD = np.std(Value)
            MEAN = np.mean(Value)
            if(Train_test):
                print("\n\nValores para o periodo de treino do Path ", i, ":")
            else:
                print("\n\nValores Finais para o Path ", i, ":")
            print("MAX:", MAX)
            print("MIN:", MIN)
            print("Mean:", MEAN)
            print("STD:", STD)
            if(save_dados):
                if(Train_test):
                    np.save('Resultados/Best_genes/Best_train_' + i +'.npy', Best_genes)
                    np.save('Resultados/Value/Val_train_'+ i +'.npy', Value)
                    np.save('Resultados/Value/Val_test_'+ i +'.npy', Value_test)
                else:
                    np.save('Resultados/Best_genes/Best_' + i +'.npy', Best_genes)
                    np.save('Resultados/Value/Val_'+ i +'.npy', Value)
            if(Train_test):
                MAX2 = np.max(Value_test)
                MIN2  = np.min(Value_test)
                STD2 = np.std(Value_test)
                MEAN2 = np.mean(Value_test)
                print("\nValores para o periodo de teste do Path ", i, ":")
                print("MAX:", MAX2)
                print("MIN:", MIN2)
                print("Mean:", MEAN2)
                print("STD:", STD2)