# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:05:15 2018

@author: kishor
"""
#import numpy as np
import matplotlib.pyplot as ptl
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data
#from sklearn.neighbors import KNeighborsClassifier
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
images_train=mnist.train.images
label_train=mnist.train.labels
images_test=mnist.test.images
label_test=mnist.test.labels
clf=svm.SVC()
clf.fit(images_train,label_train)
prediction=clf.predict(images_test)
acc=0
for i in range(0,10000):
    if prediction[i]==label_test[i]:
        acc+=1
print("Acc: ",(acc/len(label_test))*100)
