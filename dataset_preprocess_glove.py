# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:40:52 2017

@author: Akash Rastogi
"""
%reset
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pickle
import os
import pandas as pd
import gzip
import nltk
from sklearn.cross_validation import train_test_split
import random
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
import scipy
from sklearn.ensemble import *

# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

#==============================================================================
# Preprocessing the training and test dataset for GloVe
# The four main datasets to look at after prepocessing are:
# 1) X_gl_train_new_fnl; X_gl_test_new_fnl; Y_train_new, Y_test_new
# 
#==============================================================================

# Loading the dataset
data = pickle.load(open("./datasets/init_data.pickle", 'rb'))
#data = data.loc[,reviewText']]
data = data.drop('overall', 1)

# Importing GloVe dataset
def loadGloveVectors(fileid):
  f= open(fileid, 'rb')
  glove = {}
  for line in f:
    split_line = line.split()
    word = split_line[0].decode("utf-8")
    embedding = [float(val) for val in split_line[1:]]
    glove[word] = np.asarray(embedding)
  return pd.DataFrame.from_dict(glove, orient='index')
  
glove= loadGloveVectors("glove.6B.100d.txt")


X_train, X_test = data.loc[data.type == 'train','reviewText'], data.loc[data.type == 'test','reviewText']
Y_train, Y_test = data.loc[data.type == 'train','sentiment'], data.loc[data.type == 'test','sentiment']

X_train_new, Y_train_new = X_train[:1000], Y_train[:1000]
X_test_new, Y_test_new = X_test[:100], Y_test[:100]

def tokenize_df(df):
  """
  This function is to create tokens for the reviewText
  
  Input: Series of reviewText
  
  Output: dataframe with reviewText, filtered tokens and token after removing 
  stopwords
  """
  df = pd.Series.to_frame(df)
  df['word_tokens'] = '[123]'
  df['filtered_tokens'] = '[123]'
  #Tokenizing the reviewText and removing the stop words from training dataset
  for i in range(df.shape[0]):
    if i%1000 == 0:
      print(i)
    df.iloc[i, 1] = word_tokenize(df.iloc[i,0])  
    word_tokens = [w.lower() for w in df.iloc[i, 1]]
    filtered_tokens = [w for w in word_tokens if w not in stopwords.words("english")]
    df.iloc[i, 2] = filtered_tokens  
  return df

X_train_new = tokenize_df(X_train_new)
X_test_new  = tokenize_df(X_test_new)

def find_features(document):  
  '''
  To create the feature space from the given tokens for each review
  Input: Filtered_tokens: for each reviewText
  Output: features: the feature space of given review based on the top 3000 most 
  common words
  '''
  words = set(document)
  features = np.zeros(shape = 100)
  for w in words:  
    if w in glove.index:
      features += glove.loc[w,]
  return features

featuresets = [find_features(tokens) for tokens in X_train_new.filtered_tokens]
X_train_new_fnl = pd.DataFrame(featuresets)

featuresets = [find_features(tokens) for tokens in X_test_new.filtered_tokens]
X_test_new_fnl = pd.DataFrame(featuresets)

fileObj = open("./datasets/x_train_glove.pickle", "wb")
pickle.dump(X_train_new_fnl, fileObj)
fileObj.close()

fileObj = open("./datasets/x_test_glove.pickle", "wb")
pickle.dump(X_test_new_fnl, fileObj)
fileObj.close()
