# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:25:27 2017

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
from nltk.tokenize import word_tokenize
from scipy.stats import mode
import tensorflow as tf


# TO change the current working to the project folder
os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

# Importing the required datasets
f = open("./datasets/X_train_vanilla.pickle", 'rb')
X_train = pickle.load(f)
f.close()

f = open("./datasets/X_test_vanilla.pickle", 'rb')
X_test = pickle.load(f)
f.close()

f = open("./datasets/Y_train_vanilla.pickle", 'rb')
Y_train = pickle.load(f)
f.close()

f = open("./datasets/Y_test_vanilla.pickle", 'rb')
Y_test = pickle.load(f)
f.close()

# Implementing plain vanilla deep neural nets deep neural nets