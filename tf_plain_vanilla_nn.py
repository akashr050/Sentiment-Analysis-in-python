# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 22:59:14 2017

@author: Akash Rastogi
"""
#==============================================================================
# Slightly different implementation of the deep neural nets
#==============================================================================
%reset
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd
import gzip
import random
import tensorflow as tf

tb_dir = "./tb_viz/"

# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

# Importing the required datasets
f = open("./datasets/x_train_glove.pickle", 'rb')
X_train = pickle.load(f)
f.close()

f = open("./datasets/x_test_glove.pickle", 'rb')
X_test = pickle.load(f)
f.close()

f = open("./datasets/Y_train_vanilla.pickle", 'rb')
Y_train = pd.Series.to_frame(pickle.load(f))
f.close()

f = open("./datasets/Y_test_vanilla.pickle", 'rb')
Y_test = pd.Series.to_frame(pickle.load(f))
f.close()

Y_train.columns, Y_test.columns = ['pos_sent'], ['pos_sent']
Y_train['neg_sent'], Y_test['neg_sent'] = 1 - Y_train['pos_sent'], 1- Y_test['pos_sent']


def nxt_batch(X_data, Y_data, bs):
  rand_shuffle = random.sample(range(0, X_data.shape[0]), bs)
  return X_data.iloc[rand_shuffle], Y_data.iloc[rand_shuffle]

#==============================================================================
# # Implementing deep neural nets using tensorflow
#==============================================================================
n = X_train.shape[0]

# Parameters
no_epochs = 1000
batch_size = 100 
no_batch = int(n / batch_size)
no_iter = no_epochs * no_batch
alpha = 0.05 # Learning rate

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 100], name = "x")
y_ = tf.placeholder(tf.float32, [None, 2], name = "labels")

# Layers
#  tf.nn.sigmoid(
with tf.name_scope('hidden_layer'):
  W = tf.Variable(tf.truncated_normal([100, 1]), name = 'W')
  b = tf.Variable(tf.constant([0.1, 0.1]), name = 'b')
  y_logit = tf.matmul(x, W) + b

y = tf.nn.softmax(y_logit)

# Define loss and optimizer
with tf.name_scope("loss_optimizer"):
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logit))
  train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)

with tf.name_scope("predictions"):
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy of the model', accuracy)

# init
sess = tf.Session()
tb_writer = tf.summary.FileWriter(tb_dir, sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
merged = tf.summary.merge_all()
   
# Training the neural network
for i in range(no_iter):
  X_batch, Y_batch = nxt_batch(X_train, Y_train, batch_size)
  feed_dict = {x: X_batch, y_: Y_batch}
  sess.run(train_step, feed_dict= feed_dict)
  if i % no_batch == 0:
    summary_out, accur = sess.run([merged, accuracy], feed_dict = feed_dict)
    print("accuracy after ",i/no_batch," epoches is ", accur)
    tb_writer.add_summary(summary_out, i)
  

#def layer(shape):
#  W = tf.Variable(tf.truncated_normal(shape))
#  b = tf.constant(0.1, shape = shape)
#  Y = tf.matmul(x, W) + b
  
  