# -*- coding: utf-8 -*-
"""
Created on Thu May 17 01:54:40 2018

@author: kishor
"""
from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn import tree 
root = Tk()
root.title("Twitter Sentiment")
def make_label(parent, img):
    label = Label(parent, image=img,bd=10,height=150,width=150)
    label.pack(side="right")
def result():
    var.set("")
    input = text.get("1.0","end-1c")
    data=pd.read_csv('train3.csv')
    type(data)
    data.keys()
    tweet=data['tweet']
    type(tweet)
    sent_label=data['senti']
    c_vec = CountVectorizer()
    bag_of_words= c_vec.fit_transform(tweet)
    bag_of_words.shape
    len(tweet)
    type(bag_of_words)
    bag_dense= bag_of_words.todense()
    
    X_train,X_test,y_train,y_test= train_test_split(bag_dense,sent_label,test_size=0.1,
                                                    random_state=42)
    pca=PCA(n_components=2)
    pca.fit(X_train)
    new_data= pca.transform(X_train)
    clf =tree.DecisionTreeClassifier()         
    clf.fit(X_train,y_train)
    size=tweet.shape
    tweet[size[0]]=input
    sent_label=data['senti']
    c_vec = CountVectorizer()
    bag_of_words= c_vec.fit_transform(tweet)
    bag_of_words.shape
    len(tweet)
    type(bag_of_words)
    bag_dense= bag_of_words.todense()
    try:
        prediction=clf.predict(bag_dense[size[0]])
        if  prediction[0]==0:
            var.set("\nNegative tweet")
        elif prediction[0]==2:
            var.set("\nNeutral tweet")
        else:
            var.set("\nPositive tweet")
    except:
         var.set("\nWord not found in Database")


var = StringVar()
frame = Frame(root)
img = PhotoImage(file='xx.png')
make_label(frame, img)
label2=Label(frame, text="Twitter Sentiment Analysis....",fg = "skyblue",font = "Helvetica 16 bold italic")
label = Label(frame,bd=20)
text = Text(label, height=5, width=50,fg='red')


B = Button(label, text ="Sentiment", command = result ,bd=5,bg='skyblue')
label1 = Label(label, textvariable=var,fg = "red")

label2.pack()
frame.pack()
label.pack()
text.pack()
B.pack()
label1.pack()



root.mainloop()