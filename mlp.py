#------------------MULTI-LAYER PERCEPTRON------------------#

#---------------------IMPORT AREA---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------BODY------------------------#

class Perceptron():
    def __init__(self):
        self.weights = []
        self.bias = []

    def fit(self,features,labels,layers):
        #Recebe a array de features,labels e a lista contendo o numero de neuronios
        #contidos em cada layer.
        self.features = features
        self.labels = labels
        self.layers = layers
        self.weights.append(np.random.rand(self.features.shape[1],layers[0]))
        self.bias.append(np.random.rand(self.layers[0]))
        counter = 0
        for layer in self.layers[1:]:
            self.weights.append(np.random.rand(self.layers[counter],layer))
            self.bias.append(layer)
            counter+=1
    
    def ReLU(self,z, der = False):
    #Recebe um vetor de valores e executa Relu nele
    #ATENÇÃO PARA NÃO PASSAR UMA LISTA COM O ARRAY DENTRO
        saida = []
        if der == True:
            for neuron in z:
                if neuron<0:
                    saida.append(0)
                elif neuron>=0:
                    saida.append(1)
                else:
                    print("Erro na camada derivativa de ReLu")

        elif der == False:
            for neuron in z:
                if neuron<0:
                    saida.append(0)
                elif neuron>=0:
                    saida.append(neuron)
                else:
                    print('Erro na camada de ReLu')
        else:
            print("Erro critico na camada de ReLu")

        return(saida)

    def sigmoid_act(self,x, der=False):
        if (der==True): #derivative of the sigmoid
            f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
        else : # sigmoid
            f = 1/(1+ np.exp(- x))
        return f
    def train(self,lr = 0.01):
        epochs = 0
        iteration = 0
        for sample in self.features:
            self.z = []
            self.a = []
            layer = len(self.layers)-1
            for clayer in range(len(self.layers)):
                if clayer ==0:
                    self.z.append(np.dot(self.weights[clayer].transpose(),sample)+self.bias[clayer])
                    self.a.append(self.ReLU(self.z[clayer]))
                elif clayer >0 and clayer<layer:
                    backlayer = clayer-1
                    self.z.append(np.dot(self.weights[clayer].transpose(),self.a[backlayer])+self.bias[clayer])
                    self.a.append(self.ReLU(self.z[clayer]))
                elif clayer == layer:
                    backlayer = clayer-1
                    self.z.append(np.dot(self.weights[clayer].transpose(),self.a[backlayer])+self.bias[clayer])
                    self.a.append(self.sigmoid_act(self.z[clayer]))
            #BackPropagation
            target = self.labels[iteration]
            self.erro = (target-self.a[layer])**2
            self.deltas = []
            cdelta = 0 #Contador para iterar delta
            clayer = layer
            self.deltas.append(-2*(target-self.a[clayer])*self.sigmoid_act(self.z[clayer], der = True))
            while clayer >0:
                self.deltas.append(np.dot(self.deltas[cdelta],self.weights[clayer].transpose())*self.ReLU(self.z[clayer-1],der=True))
                clayer-=1
                cdelta+=1
            cdelta = 0
            clayer = layer
            while clayer >=0: 
                if clayer ==0:
                    self.weights[clayer] = self.weights[clayer]-(lr*np.kron(self.deltas[cdelta],sample).reshape(self.weights[clayer].shape))
                    self.bias[clayer] = self.bias[clayer] - lr*self.deltas[cdelta]

                else:
                    self.weights[clayer]= self.weights[clayer]-(lr*np.kron(self.deltas[cdelta],self.a[clayer-1]).reshape(self.weights[clayer].shape))
                    self.bias[clayer] = self.bias[clayer] - lr*self.deltas[cdelta]
                
                cdelta +=1
                clayer -=1
            iteration+=1
            print(self.deltas)


    def result(self):
        self.predictions = []
        for sample in self.features:
            self.z = []
            self.a = []
            layer = len(self.layers)-1
            for clayer in range(len(self.layers)):
                if clayer ==0:
                    self.z.append(np.dot(self.weights[clayer].transpose(),sample)+self.bias[clayer])
                    self.a.append(self.ReLU(self.z[clayer]))
                elif clayer >0 and clayer<layer:
                    backlayer = clayer-1
                    self.z.append(np.dot(self.weights[clayer].transpose(),self.a[backlayer])+self.bias[clayer])
                    self.a.append(self.ReLU(self.z[clayer]))
                elif clayer == layer:
                    backlayer = clayer-1
                    self.z.append(np.dot(self.weights[clayer].transpose(),self.a[backlayer])+self.bias[clayer])
                    self.a.append(self.sigmoid_act(self.z[clayer]))

            if self.a[layer][0]>=0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        self.predictions = np.array(self.predictions)
        self.resultado = pd.DataFrame(self.predictions,self.labels)
        self.resultado.to_csv('resultado.csv')
        print(self.resultado)