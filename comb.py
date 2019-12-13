    # -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:20:53 2018

@author: thiago
"""
import numpy as np

combinacoes = []
for p1 in range(1,12):
    for p2 in range(p1+1,13):
        for p3 in range(p2+1,14):
            for p4 in range(p3+1,15):
                for p5 in range(p4+1,16):
                    for p6 in range(p5+1,17):
                        for p7 in range(p6+1,18):
                            for p8 in range(p7+1,19):
                                for p9 in range(p8+1,20):
                                    for p10 in range(p9+1,21):
                                        for p11 in range(p10+1,22):
                                            for p12 in range(p11+1,23):
                                                for p13 in range(p12+1,24):
                                                    for p14 in range(p13+1,25):
                                                        for p15 in range(p14+1,26):
                                                            combinacoes.append([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15])



combinacoes= np.array(combinacoes) 

combinacoesP = []
for i in combinacoes: 
    inserir = True
    qtdPar = 0
    qtdImpar = 0
    qtdPrimos = 0
    provavel = True
    for j in i:
        if j in [5,6,7,8,15,20,11]:
            provavel = False
        if j in [2,3,5,7,11,13,17,19,23]:
            qtdPrimos += 1
        if j%2 == 1:
            qtdImpar += 1
        if j%2 == 0:
            qtdPar += 1
        if (qtdPar == 7 and qtdImpar == 8) and qtdPrimos >= 4 and qtdPrimos <= 6 and provavel:
            combinacoesP.append(i)
        
combinacoesP = np.array(combinacoesP)

combinacoesP2 = []
for i in combinacoesP:
    #i = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]           
    contadorSequencia = i[0]
    repeticoes = 0
    inserir = True
    intervalos = 0
    for j in i:
        if (contadorSequencia == j and repeticoes >= 5) :
            inserir = False
            break       
        elif contadorSequencia == j:
            repeticoes += 1
        elif contadorSequencia != j:
            intervalos += 1
            contadorSequencia = j
            repeticoes = 0
        contadorSequencia += 1
    if inserir and intervalos >= 4 and intervalos < 6:
        #print(intervalos)
        #i = np.append(i,intervalos)
        combinacoesP2.append(i)

combinacoesP2 = np.array(combinacoesP2)

ganhos = []
for concurso in range(1696,1729):
    acertos = []
    for i in combinacoesP2:
        acert = 0
        for x in i:  
            if x in resultados[concurso]:
                acert += 1
        if acert == 15:
            print i
        acertos.append(acert)    
    ganho = 0
    for i in acertos:
        if i == 11:
            ganho += 4
        if i == 12:
            ganho += 8
        if i == 13:
            ganho += 20
        if i == 14:
            ganho += 1000
        if i == 15:
            print i
            ganho += 1000000
    ganhos.append(ganho)
    
sum(ganhos)

232*2*35
152950

1214412

    

    


    

