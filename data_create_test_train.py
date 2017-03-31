# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:52:42 2017

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


# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

# Checking the working directory
os.getcwd()

# Writing code to extract the required dataset
# Amazon review dataset is skewed so kinda bootstrapped to change the positive/
# negative sentiment ratio to 0.5

def parse(path):
  """
  This function is used to parse the amazon data set downloaded from 
  http://jmcauley.ucsd.edu/data/amazon/
  """
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)


def getDF(path, type_review):
  """
  This function is used to get the dataset for each kind of sentiment attached 
  with a review. We are extracting only 50K reviews attached with each sentiment
  """
  i = 0
  df = {}
  for d in parse(path):
    if type_review == 'positive':  
      if d['overall'] >= 5:
        d['sentiment'] = 1.0
        df[i] = d
        i += 1
    elif type_review == 'negative':
      if d['overall'] <= 2:  
        d['sentiment'] = 0.0
        df[i] = d
        i += 1      
    if i > 50000:
      break
  return pd.DataFrame.from_dict(df, orient='index')

df_positive = getDF('reviews_Books_5.json.gz', "positive")
df_negative = getDF('reviews_Books_5.json.gz', "negative")

# Cross checking the unique values in df_negative
df_negative.sentiment.unique()
df_positive.sentiment.unique()
# Selecting the desired columns

df = pd.concat([df_negative, df_positive])
df_1 = df.loc[:,['sentiment', 'overall', 'reviewText', 'summary']]

X = df_1.iloc[:,1:3]
Y = df_1.iloc[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, 
                                                    test_size = 0.2)
X_test['type'], X_train['type'] = 'test', 'train'
train = pd.concat([X_train, Y_train], axis = 1)
test = pd.concat([X_test, Y_test], axis = 1)
fnl_data = pd.concat([train, test], axis = 0)

fileObj = open("init_data.pickle", "wb")
pickle.dump(fnl_data, fileObj)
fileObj.close()
