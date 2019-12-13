import numpy as np
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

resultados = pd.read_csv('resultadosvirgulas.csv')
resultados = resultados[::-1]

numeros = resultados.iloc[:,9:24].values


concursos = resultados.iloc[:,0:1].values
concursos = np.array([x % 2 for x in concursos])
vencedor15 = resultados.iloc[:,4:5].values
vencedor15 = np.array([1 if x[0] > 1 else 0 for x in vencedor15])
semGanhadores = resultados.iloc[:,4:5].values
semGanhadores = np.array([1 if x[0] == 0 else 0 for x in semGanhadores])
local = resultados.iloc[:,2:3].values
local = np.array([1 if "CAMINH" in x[0] else 0 for x in local])
epocaano = resultados.iloc[:,1:2].values
epocaano = np.array([1 if int(x[0].split('/')[1].split('/')[0]) > 6 else 0 for x in epocaano])
epocames = resultados.iloc[:,1:2].values
epocames = np.array([1 if int(x[0].split('/')[0]) > 15 else 0 for x in epocames])

numeros_sorteados = []
for sorteio in numeros: 
    nums = []
    for i in range(1,26):
        if i in sorteio:
            nums.append(1)
        else:
            nums.append(0)
    numeros_sorteados.append(nums)  

#
numerosBinario = numeros_sorteados

for i in range(0, 1695):   
    numerosBinario[i].append(vencedor15[i])
    numerosBinario[i].append(semGanhadores[i])
    numerosBinario[i].append(local[i])
    #numerosBinario[i].append(concursos[i][0])
    if i < 1694:
        numerosBinario[i].append(concursos[i+1][0])
        numerosBinario[i].append(epocaano[i+1])
        numerosBinario[i].append(epocames[i+1])
    else:
        numerosBinario[i].append(1)
        numerosBinario[i].append(1)
        numerosBinario[i].append(0)
        
numerosBinario = np.array(numerosBinario)    
    

previsores = []
num_real = []
#num1_real = []
#num2_real = []

quantidadeAmostrasAnteriores = 10
quantidadePrevisores = 25
for i in range(quantidadeAmostrasAnteriores, 1695):
    previsores.append(numerosBinario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
    num_real.append(numerosBinario[i, 0:25])
    #num2_real.append(numerosBinario[i, 1])
previsores, num_real = np.array(previsores), np.array(num_real)    
#previsores, num1_real,num2_real = np.array(previsores), np.array(num1_real), np.array(num2_real) 
#num_real = np.column_stack((num1_real,num2_real))




regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], quantidadePrevisores)))
regressor.add(Dropout(0.3))

#regressor.add(LSTM(units = 400, return_sequences = True, activation = 'sigmoid'))
#regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))

#regressor.add(LSTM(units = 300, return_sequences = True, activation = 'sigmoid'))
#regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 25, activation = 'linear'))   


regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['binary_accuracy'])


es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 25, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1)


#regressor.fit(previsores, num_real, epochs = 150, batch_size = 32,
#              callbacks = [es, rlr, mcp])

regressor.fit(previsores, num_real, epochs = 350, batch_size = 32)
    

X_teste = []
for i in range(1695, 1696):
    print i
    X_teste.append(numerosBinario[i-quantidadeAmostrasAnteriores:i, 0:quantidadePrevisores])
X_teste = np.array(X_teste)
previsoes = regressor.predict(X_teste)


valores_reais = []
for i in range(1694, 1695):
    valores_reais.append(numeros[i])

#previsoes = normalizador.inverse_transform(previsoes)
#X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
        
        
    
