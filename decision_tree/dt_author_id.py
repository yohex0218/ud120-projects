#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
from time import time
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


# features_train and features_test are the features for the training
# and testing data sets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# your code goes here

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

print "training data size:", len(features_train), len(features_train[0])
print "test data size:", len(features_test)

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"

t0 = time()
prediction = clf.predict(features_test)

print "predicting time:", round(time()-t0, 3), "s"

print "Accuracy:", accuracy_score(labels_test, prediction), "%"
