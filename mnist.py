#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:23:49 2023

@author: fibonacci
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

datasets = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = datasets.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=50)
model.evaluate(x_test,y_test)
# Plot training history
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
