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
import numpy as np
from PIL import Image

dataset=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=dataset.load_data()
x_train,x_test=x_train/255.0,x_test/255.0


class_label=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal',
             'Shirt','Sneaker','Bag','Ankle boot']

def image_display(image_position):
    plt.Figure(figsize=(8,4))
    plt.imshow(x_train[image_position])
    plt.xlabel(class_label[y_train[image_position]])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
image_display(43)

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
    ])
callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x_train,y_train,batch_size=32,epochs=100,validation_split=0.25,callbacks=[callbacks])
test_loss,test_accuracy=model.evaluate(x_test,y_test,verbose=2)
print("Accuracy:",test_accuracy)
#plt.plot(history.history['accuracy'])

#single prediction
image = Image.open('/home/fibonacci/Documents/hs.jpg').convert('L')
image = image.resize((28, 28))
image = np.array(image) / 255.0
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_class = np.argmax(prediction)

print('Predicted class:', class_label[predicted_class])