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
from nltk.tokenize import word_tokenize
import scipy
from sklearn.ensemble import *
from sklearn.ensemble import GradientBoostClassifier

# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

#==============================================================================
# Preprocessing the training and test dataset for traditional techniques
# The four main datasets to look at after prepocessing are:
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
#tokenizer = RegexpTokenizer(r'\w+')
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


#==============================================================================
# Implementing the various techniques on the test and train datasets
# 
#==============================================================================

# Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb_model = gnb.fit(X_train_new_fnl, Y_train_new)
gnb_model_predictions = gnb_model.predict(X_test_new_fnl)
gnb_model_accuracy = sum(gnb_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("gnb_vanilla.pickle", "wb")
pickle.dump(gnb_model, save_classifier)
save_classifier.close()



# Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb_model = mnb.fit(X_train_new_fnl, Y_train_new)
mnb_model_predictions = mnb_model.predict(X_test_new_fnl)
mnb_model_accuracy = sum(mnb_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("mnb_vanilla.pickle", "wb")
pickle.dump(mnb_model, save_classifier)
save_classifier.close()


# Bernoulli Naive Bayes Classifier
bnb = BernoulliNB()
bnb_model = bnb.fit(X_train_new_fnl, Y_train_new)
bnb_model_predictions = bnb_model.predict(X_test_new_fnl)
bnb_model_accuracy = sum(bnb_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("bnb_vanilla.pickle", "wb")
pickle.dump(bnb_model, save_classifier)
save_classifier.close()

# Logistic Regression
lr = LogisticRegression()        
lr_model = lr.fit(X_train_new_fnl, Y_train_new)
lr_model_predictions = lr_model.predict(X_test_new_fnl)
lr_model_accuracy = sum(lr_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("lr_vanilla.pickle", "wb")
pickle.dump(lr_model, save_classifier)
save_classifier.close()

# LinearSVM
lsvm = SVC(kernel = 'linear', C=1)        
lsvm_model = lsvm.fit(X_train_new_fnl, Y_train_new)
lsvm_model_predictions = lsvm_model.predict(X_test_new_fnl)
lsvm_model_accuracy = sum(lsvm_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("lsvm_vanilla.pickle", "wb")
pickle.dump(lsvm_model, save_classifier)
save_classifier.close()

# rbfSVM
rsvm = SVC(kernel = 'rbf', C=1, gamma = 0.01)        
rsvm_model = rsvm.fit(X_train_new_fnl, Y_train_new)
rsvm_model_predictions = rsvm_model.predict(X_test_new_fnl)
rsvm_model_accuracy = sum(rsvm_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("rsvm_vanilla.pickle", "wb")
pickle.dump(rsvm_model, save_classifier)
save_classifier.close()

# polySVM
psvm = SVC(kernel = 'poly', C=1, gamma = 0.1, degree = 2)
psvm_model = psvm.fit(X_train_new_fnl, Y_train_new)
psvm_model_predictions = psvm_model.predict(X_test_new_fnl)
psvm_model_accuracy = sum(psvm_model_predictions == Y_test_new) / Y_test_new.shape[0]

save_classifier = open("psvm_vanilla.pickle", "wb")
pickle.dump(psvm_model, save_classifier)
save_classifier.close()

# Dataset creation for 

        
        
        
        
        
        
        
        
        
        
    
    
    

