#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:44:11 2018

@author: thiagocrestani
"""

import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

quantidadeAmostrasAnteriores = 8
quantidadePrevisores = 25

resultados_treinamento = pd.read_csv('resultadosvirgulas.csv')[::-1]
resultados_teste = pd.read_csv('resultados_teste.csv')[::-1]

resultados = np.concatenate((resultados_treinamento.iloc[:,9:24].values,resultados_teste.iloc[:,1:16].values))


numeros_binario = []
for item in resultados: 
    nums = []
    for i in range(1,26):
        if i in item:
            nums.append(1)
        else:
            nums.append(0)
    numeros_binario.append(nums)
    
numeros_binario = np.array(numeros_binario) 


quantidadeAmostrasAnteriores = 4
quantidadePrevisores = 25
previsores = []
for i in range(quantidadeAmostrasAnteriores, 1695):
    previsores.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    #num_real.append(numerosBinario[i, 0:25])

previsores = np.array(previsores)




regressor = Sequential()
regressor.add(LSTM(units = 900, return_sequences = True, input_shape = (previsores.shape[1], quantidadePrevisores)))
regressor.add(Dropout(0.3))

#regressor.add(LSTM(units = 400, return_sequences = True, activation = 'sigmoid'))
#regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 900, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 700, return_sequences = True))
regressor.add(Dropout(0.3))

#regressor.add(LSTM(units = 300, return_sequences = True, activation = 'sigmoid'))
#regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 700, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 500, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 400, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 400, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 300, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 300, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 200))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 25, activation = 'linear'))   


regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['binary_accuracy'])


regressor.load_weights('pesos5_treinoLongo.h5')


vacertos = []
for concurso in range(50,1729):
    temp = [concurso]
    vacertos.append(temp)


#mais provaveis
a = 0    
for concurso in range(50,1729):
    concurso = 1699
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    for i in range(0,10):
        res.remove(min(res))   
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0)   
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1       
    vacertos[a].append(acertos)
    a += 1
    #print acertos
    
#menos provaveis  
a = 0 
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    for i in range(0,10):
        res.remove(max(res))
    
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0)  
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1        
    #print acertos 
  

    
#boa    
#meio
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    #temp = []
    
    res = list(previsoes.ravel())   
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    
        
    #res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1 

#boa   
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    #temp = []
    
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    
    
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    
        
    #res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1 

#boa
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    #temp = []
    
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))

    
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    
        
    #res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1   
  
    
    
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []    
    temp.append(min(res))  
    res.remove(min(res))
    temp.append(min(res))  
    res.remove(min(res))
    temp.append(min(res))  
    res.remove(min(res))
    temp.append(min(res))  
    res.remove(min(res))
    
    temp.append(max(res))  
    res.remove(max(res))
    temp.append(max(res))  
    res.remove(max(res))
    temp.append(max(res))  
    res.remove(max(res))
    temp.append(max(res))  
    res.remove(max(res))
    temp.append(max(res))  
    res.remove(max(res))
    
    
    
    
    
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
   

    
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
  
    
        
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1

#verificar
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(min(res)) 
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
        
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1 

a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
     
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
        
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1     


#melhor de todos
a = 0     
for concurso in range(50,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    
    
    
        
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1 


a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
           
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1


#nao
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(min(res))
    res.remove(min(res))
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))        
    temp.append(min(res))
    res.remove(min(res))       
    res.remove(min(res))
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))        
    temp.append(min(res))
    res.remove(min(res))       
    res.remove(min(res))
    res.remove(min(res))
    
    temp.append(min(res))  
    res.remove(min(res))        
    temp.append(min(res))
    res.remove(min(res))       
    res.remove(min(res))
    res.remove(min(res))
       
    temp.append(min(res))  
    res.remove(min(res))    
    res.remove(min(res))
           
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1      
 
    
#boa tb    
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(max(res))
    res.remove(max(res))
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
            
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1
    
a = 0     
for concurso in range(1696,1729):
    #concurso = 1702
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    
    temp = []         
    res.remove(max(res))
    res.remove(max(res))
    
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
    
    temp.append(max(res))  
    res.remove(max(res))    
    res.remove(max(res))
         
    res = temp + res 
    apostas = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            apostas.append(1)
        else:
            apostas.append(0) 
    acertos = 0
    #concurso = 1696
    for i in range(0,25):
        if apostas[i] == 1 and numeros_binario[concurso][i] == 1:
            acertos += 1
    vacertos[a].append(acertos)
    a += 1    
    

for i in range(0,len(vacertos)):
    print vacertos[i]
    
    
    
a = 0 
retirarTodos = []   
for concurso in range(1696,1729):
    #concurso = 1699
    X_teste = []
    for i in range(concurso-1, concurso):
        X_teste.append(numeros_binario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    previsoes = regressor.predict(np.array(X_teste))
    #conferir
    res = list(previsoes.ravel())
    for i in range(0,18):
        res.remove(max(res))   
    retirar = []
    for i in range(0,25):
        if previsoes[0][i] in res:
            retirar.append(i)
    retirarTodos.append(retirar)