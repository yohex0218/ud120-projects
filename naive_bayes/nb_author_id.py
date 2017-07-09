#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""


from time import time
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# features_train and features_test are the features for the training
# and testing data sets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print "training data size:", len(features_train)
print "test data size:", len(features_test)

t0 = time()
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"

t0 = time()
prediction = classifier.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

print "Accuracy:", accuracy_score(labels_test, prediction), "%"
