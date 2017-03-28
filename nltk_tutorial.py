# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:57:19 2017

@author: Akash Rastogi
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle
import os
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

os.chdir("C:/Users/Akash Rastogi/Desktop/UMich_study_material/winter_2017/EECS_597/project")

documents = [(list(movie_reviews.words(fileid)), category) 
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents) 
all_words = [w.lower() for w in movie_reviews.words()]
freq_words = nltk.FreqDist(all_words)
word_features = list(freq_words.keys())[:3000] # Don't agree that these are the top 3000 words.
# Have to check again

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# TO check the most common words existing in your document
# print([k for k in featuresets[0][0].keys() if featuresets[0][0][k] == True])        
        
train_set = featuresets[:1900]
test_set  = featuresets[1900:]       

classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(15) 

save_classifier = open("Naive_bayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
        
classifier_f = open("Naive_bayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
        
classifier.show_most_informative_features(15)        

# Multinomial NB classifier        
MNB_classifier = SklearnClassifier(MultinomialNB())        
MNB_classifier.train(train_set)
nltk.classify.accuracy(MNB_classifier, test_set)        

# Bernoulli NB classifier        
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())        
BernoulliNB_classifier.train(train_set)
nltk.classify.accuracy(BernoulliNB_classifier, test_set)        


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())        
LogisticRegression_classifier.train(train_set)
nltk.classify.accuracy(LogisticRegression_classifier, test_set)        

LinearSVC_classifier = SklearnClassifier(LinearSVC())        
LinearSVC_classifier.train(train_set)
nltk.classify.accuracy(LinearSVC_classifier, test_set)        

# Have to watch last three tutorials as well         
        
        
        
        
        
        
        
        
        
        
        
    
    
    