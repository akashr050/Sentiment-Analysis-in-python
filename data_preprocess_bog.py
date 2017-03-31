# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:01:47 2017

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
from nltk.tokenize import word_tokenize, RegexpTokenizer
import scipy
from sklearn.ensemble import *
from sklearn.ensemble import GradientBoostClassifier
from nltk.stem import WordNetLemmatizer

# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

#==============================================================================
# Preprocessing the training and test dataset for traditional feature creation
# based on the frequency of the most common words
# The steps involved in the process are:
# 1) Tokenize the reviews
# 2) Remove the stop words
# 3) Choose the top 3000 words among all the words in corpus
# 4) Create hot representations for all reviews based on these 3000 words 

# Parameters: 
# Things we want to remove i.e stop words, punctuations
# Number of features currently 3000
#  
# The four main output datasets to look at after prepocessing are:
# 1) X_train_new_fnl; X_test_new_fnl; Y_train_new, Y_test_new
# 
#==============================================================================

# Loading the dataset
data = pickle.load(open("init_data.pickle", 'rb'))
#data = data.loc[,reviewText']]
data = data.drop('overall', 1)

X_train, X_test = data.loc[data.type == 'train','reviewText'], data.loc[data.type == 'test','reviewText']
Y_train, Y_test = data.loc[data.type == 'train','sentiment'], data.loc[data.type == 'test','sentiment']

X_train_new, Y_train_new = X_train[:1000], Y_train[:1000]
X_test_new, Y_test_new = X_test[:100], Y_test[:100]


X_train_new = pd.Series.to_frame(X_train_new)
X_train_new['word_tokens'] = '[123]'
X_train_new['filtered_tokens'] = '[123]'


all_words = []
# tokenizer = RegexpTokenizer(r'\w+')
#Tokenizing the reviewText and removing the stop words from training dataset
for i in range(X_train_new.shape[0]):
  X_train_new.iloc[i, 1] = word_tokenize(X_train_new.iloc[i,0])  
  word_tokens = [w.lower() for w in X_train_new.iloc[i, 1]]
  filtered_tokens = [w for w in word_tokens if w not in stopwords.words("english")]
  X_train_new.iloc[i, 2] = filtered_tokens  
  all_words += filtered_tokens
  if i%1000 == 0:
    print(i)


X_test_new = pd.Series.to_frame(X_test_new)
X_test_new['word_tokens'] = '[123]'
X_test_new['filtered_tokens'] = '[123]'


#Tokenizing the reviewText and removing the stop words from test dataset
for i in range(X_test_new.shape[0]):
  X_test_new.iloc[i, 1] = word_tokenize(X_test_new.iloc[i,0])  
  word_tokens = [w.lower() for w in X_test_new.iloc[i, 1]]
  filtered_tokens = [w for w in word_tokens if w not in stopwords.words("english")]
  X_test_new.iloc[i, 2] = filtered_tokens  
  if i%1000 == 0:
    print(i)

#Creating the feature space of top 3000 words for each token for both test and 
#train dataset
num_of_features = 3000
freq_words = nltk.FreqDist(all_words)
freq_words = pd.DataFrame(freq_words.most_common(num_of_features))
word_features = list(freq_words[0])

def find_features(document):  
  '''
  To create the feature space from the given tokens for each review
  Input: Filtered_tokens: for each reviewText
  Output: features: the feature space of given review based on the top 3000 most 
  common words
  '''
  words = set(document)
  features = {}
  for w in word_features:
      features[w] = (w in words)
  return features


featuresets = [find_features(tokens) for tokens in X_train_new.filtered_tokens]
X_train_new_fnl = pd.DataFrame(featuresets)

featuresets = [find_features(tokens) for tokens in X_test_new.filtered_tokens]
X_test_new_fnl = pd.DataFrame(featuresets)

save_data = open("X_train_vanilla.pickle", "wb")
pickle.dump(X_train_new_fnl, save_data)
save_data.close()

save_data = open("X_test_vanilla.pickle", "wb")
pickle.dump(X_test_new_fnl, save_data)
save_data.close()

save_data = open("Y_train_vanilla.pickle", "wb")
pickle.dump(Y_train_new, save_data)
save_data.close()

save_data = open("Y_test_vanilla.pickle", "wb")
pickle.dump(Y_test_new, save_data)
save_data.close()

        
        
        
        
        
        
        
        
        
        
    
    
    

