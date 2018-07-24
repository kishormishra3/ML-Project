# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 06:35:04 2018

@author: kishor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as ptl
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
data=pd.read_csv('gender.csv')
arr=np.array(data)
label=np.array(data['gender'])
label=label[:3208]
a=np.zeros((3208,10000))
for i in range(0,3208):
    m=0
    for j in range(3,10003):
        a[i][m]=arr[i][j]
        m+=1
ptl.imshow(a[1].reshape(100,100))
np.save('data.npy',a)
np.save('label.npy',label)
images = np.array([i.reshape((100,100))for i in a])
for i in range (0,3208):
    cv2.imwrite('images/img{:>0}.jpg'.format(i),images[i])
label=np.load('label.npy')
images=np.load('data.npy')
images_flat = np.array([i.reshape((10000,))for i in images])
x_train,x_test,y_train,y_test= train_test_split(images_flat,label,test_size=0.3,
                                                random_state=0)
pca=PCA(n_components=600)
pca.fit(x_train)
x_train_= pca.transform(x_train)
pca=PCA(n_components=600)
pca.fit(x_test)
x_test_= pca.transform(x_test)
clf=svm.SVC()
clf.fit(x_train_,y_train)
prediction=clf.predict(x_test_)
pred_labels = prediction == y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('SVM accuracy=',(acc)/len(pred_labels)*100,'%')