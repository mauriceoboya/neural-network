import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt


dataset=pd.read_csv('diabetes.csv')


x=[2,4,5,6]
weights=[[0.34,0.23,-0.56,0.9],
         [0.12,0.98,-0.12,0.12],
         [0.56,0.43,0.04,.98]]
bias=2

Outcome=nm.dot(weights,x)+bias
print(Outcome)