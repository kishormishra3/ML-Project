# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 14:04:41 2018

@author: kishor
"""


import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv('train3.csv')
type(data)
data.keys()
tweet=data['tweet']
type(tweet)
tweet.shape
tweet[0]
tweet[0:10]




sent_label=data['senti']
c_vec = CountVectorizer()
bag_of_words= c_vec.fit_transform(tweet)
bag_of_words.shape
len(tweet)
type(bag_of_words)
bag_dense=bag = bag_of_words.todense()
bag_dense.shape
bag_dense





X_train,X_test,y_train,y_test= train_test_split(bag_dense,sent_label,test_size=0.1,
                                                random_state=42)
pca=PCA(n_components=2)
pca.fit(X_train)
new_data= pca.transform(X_train)
clf=svm.SVC()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
pred_labels = prediction == y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('SVM accuracy=',(acc)/len(pred_labels)*100,'%')



clf =KNeighborsClassifier()         
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
pred_labels=prediction==y_test
acc=0.0
for i in pred_labels:
    if i==True:
        acc+=1
print ('KNN accuracy=',(acc)/len(pred_labels)*100,'%')