#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

# add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# your code goes here
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, random_state=42, test_size=.3)

print len(features), len(features_train), len(features_test)
print len(labels), len(labels_train), len(labels_test)

classifier = tree.DecisionTreeClassifier().fit(features_train, labels_train)
prediction = classifier.predict(features_test)

print prediction, sum(prediction)
print labels_test, sum(labels_test)
print prediction + labels_test

print "Accuracy : ", accuracy_score(labels_test, prediction)

