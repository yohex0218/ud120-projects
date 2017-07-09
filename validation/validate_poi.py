#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


# it's all yours from here forward!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, random_state=42, test_size=.3)

print len(features), len(features_train), len(features_test)
print len(labels), len(labels_train), len(labels_test)

classifier = tree.DecisionTreeClassifier().fit(features_train, labels_train)
prediction = classifier.predict(features_test)

print "Accuracy : ", accuracy_score(labels_test, prediction)



