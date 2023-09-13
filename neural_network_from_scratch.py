import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
nm.random.seed(0)

dataset=pd.read_csv('diabetes.csv')
x1=dataset.drop(columns=['Outcome'],axis=1)
y=dataset['Outcome']
str=StandardScaler()
x1=str.fit_transform(x1)


class Forwardpass:
    def __init__(self, hidden_inputs, hidden_neurons):
        self.weight = nm.random.rand(hidden_inputs,hidden_neurons)
        self.biases=nm.zeros((1,hidden_neurons))
    def outcome(self,inputs):
        self.output=nm.dot(inputs,self.weight)+self.biases
        

class Activation_Function:
    def forward(self,input):
        self.output=nm.maximum(0,input)

layer1=Forwardpass(8,3)
activation1=Activation_Function()
layer2=Forwardpass(3,4)
activation2=Activation_Function()
layer3=Forwardpass(4,2)
layer1.outcome(x1)
activation1.forward(layer1.output)
layer2.outcome(activation1.output)
print(layer2.output)

activation2.forward(layer2.output)
layer3.outcome(activation2.output)
print(layer3.output)
