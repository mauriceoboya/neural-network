#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 05:00:38 2023

@author: fibonacci
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt

dataset=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=dataset.load_data()
x_train,x_test=x_train/255.0,x_test/255.0


class_label=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal',
             'Shirt','Sneaker','Bag','Anklle boot']

plt.Figure()
plt.imshow(x_train[8])
plt.colorbar()
plt.grid(False)
plt.show()