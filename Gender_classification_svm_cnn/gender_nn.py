# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 08:10:04 2018

@author: kishor
"""

import keras
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
label=np.load('label.npy')
images=np.load('data.npy')
images_flat = np.array([i.reshape((10000,))for i in images])
one_hot=keras.utils.to_categorical(label)
inp=Input(shape=(10000,),name='inp')
hid=Dense(10,activation='sigmoid')(inp)
out=Dense(2,activation='sigmoid')(hid)
model=Model([inp],out)
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(images_flat,one_hot,epochs=10,verbose=2)
'''
images_ = np.array([i.reshape((100,100))for i in images])
x_train = images_.reshape(images_.shape[0], 100, 100, 1)
inp=Input(shape=(100,100,1))
conv=Conv2D(32,(3,3),activation='relu')(inp)
pool=MaxPooling2D((2,2))(conv)
flat=Flatten()(pool)
hid=Dense(100,activation='sigmoid')(flat)
out=Dense(2,activation='sigmoid')(hid)
model = Model([inp],out)
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit(x_train,one_hot,epochs=10,verbose=2)
'''