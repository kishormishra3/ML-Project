# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:14:22 2017

@author: kishor
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,shape=[None,784])
y_=tf.placeholder(tf.float32,shape=[None,10])
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w_convl =weight_variable([5,5,1,32])
b_convl =bias_variable([32])
x_image =tf.reshape(x,[-1,20,20,1])



sess.run(tf.global_variables_initializer())
cross_entropy=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)
)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch= data.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y:batch[1]})
correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval(feed_dict={x:data.test.images,y:data.test.labels}))