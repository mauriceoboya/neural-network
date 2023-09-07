import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


input=[1,2,3]
weights=[0.2,0.8,-0.5]
bias=0.2

output=input[0]*weights[0]+input[1]*weights[1]+input[2]*weights[2]+bias
print(output)