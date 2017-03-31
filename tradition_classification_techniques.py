# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:13:34 2017

@author: Akash Rastogi
"""
#==============================================================================
# Implementing the various traditional classification techniques on the 
# test and train datasets
# The four imput datasets are X_train, Y_train, X_test, Y_test 
# The various models implemented in this module:
# 1) Gaussian Naive Bayes Classifier
# 2) Multinomial Naive Bayes Classifier
# 3) Bernoulli Naive Bayes Classifier
# 4) Logistic Regression
# 5) LinearSVM
# 6) rbfSVM
# 7) polySVM
# 8) ensemble model of all the classifiers
#
#==============================================================================

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
from scipy.stats import mode

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

# Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb_model = gnb.fit(X_train, Y_train)
gnb_model_predictions = gnb_model.predict(X_test)
gnb_model_accuracy = sum(gnb_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("gnb_vanilla.pickle", "wb")
pickle.dump(gnb_model, save_classifier)
save_classifier.close()



# Multinomial Naive Bayes Classifier
mnb = MultinomialNB()
mnb_model = mnb.fit(X_train, Y_train)
mnb_model_predictions = mnb_model.predict(X_test)
mnb_model_accuracy = sum(mnb_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("mnb_vanilla.pickle", "wb")
pickle.dump(mnb_model, save_classifier)
save_classifier.close()


# Bernoulli Naive Bayes Classifier
bnb = BernoulliNB()
bnb_model = bnb.fit(X_train, Y_train)
bnb_model_predictions = bnb_model.predict(X_test)
bnb_model_accuracy = sum(bnb_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("bnb_vanilla.pickle", "wb")
pickle.dump(bnb_model, save_classifier)
save_classifier.close()

# Logistic Regression
lr = LogisticRegression()        
lr_model = lr.fit(X_train, Y_train)
lr_model_predictions = lr_model.predict(X_test)
lr_model_accuracy = sum(lr_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("lr_vanilla.pickle", "wb")
pickle.dump(lr_model, save_classifier)
save_classifier.close()

# LinearSVM
lsvm = SVC(kernel = 'linear', C=1)        
lsvm_model = lsvm.fit(X_train, Y_train)
lsvm_model_predictions = lsvm_model.predict(X_test)
lsvm_model_accuracy = sum(lsvm_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("lsvm_vanilla.pickle", "wb")
pickle.dump(lsvm_model, save_classifier)
save_classifier.close()

# rbfSVM
rsvm = SVC(kernel = 'rbf', C=1, gamma = 0.01)        
rsvm_model = rsvm.fit(X_train, Y_train)
rsvm_model_predictions = rsvm_model.predict(X_test)
rsvm_model_accuracy = sum(rsvm_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("rsvm_vanilla.pickle", "wb")
pickle.dump(rsvm_model, save_classifier)
save_classifier.close()

# polySVM
psvm = SVC(kernel = 'poly', C=1, gamma = 0.1, degree = 2)
psvm_model = psvm.fit(X_train, Y_train)
psvm_model_predictions = psvm_model.predict(X_test)
psvm_model_accuracy = sum(psvm_model_predictions == Y_test) / Y_test.shape[0]

save_classifier = open("psvm_vanilla.pickle", "wb")
pickle.dump(psvm_model, save_classifier)
save_classifier.close()

# Ensemble model of all the models
ensemble_pred = pd.DataFrame(dict(gnb = gnb_model_predictions,
                             mnb = mnb_model_predictions,
                             bnb = bnb_model_predictions,
                             lr = lr_model_predictions,
                             lsvm = lsvm_model_predictions,
                             rsvm = rsvm_model_predictions,
                             psvm = psvm_model_predictions))

ensemble_predictions = ensemble_pred.mode(1)
ensemble_accuracy = sum(ensemble_predictions.iloc[:,0] == Y_test) / Y_test.shape[0]
