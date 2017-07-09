#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary
"""


import random
import numpy as np
import pylab as pl
from sklearn import ensemble
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#
#  make the toy data set
#
def make_terrain_data(n_points=1000):
    random.seed(42)
    grade = [random.random() for ii in range(0, n_points)]
    bumpy = [random.random() for ii in range(0, n_points)]
    error = [random.random() for ii in range(0, n_points)]

    x = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0, n_points)]

    for ii in range(0, len(y)):
        if grade[ii] > 0.8 or bumpy[ii] > 0.8:
            y[ii] = 1.0

    # split into train/test sets
    split = int(0.75*n_points)
    x_train = x[0:split]
    x_test = x[split:]
    y_train = y[0:split]
    y_test = y[split:]

    return x_train, y_train, x_test, y_test


#
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
#
def create_graph(classifier, x_test, y_test):
    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    bumpy_sig = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 0]
    grade_bkg = [x_test[ii][0] for ii in range(0, len(x_test)) if y_test[ii] == 1]
    bumpy_bkg = [x_test[ii][1] for ii in range(0, len(x_test)) if y_test[ii] == 1]

    plt.scatter(grade_sig, bumpy_sig, color="b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("output_adaboost.png")


features_train, labels_train, features_test, labels_test = make_terrain_data()

clf = ensemble.AdaBoostClassifier()

clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)

print "Test Accuracy : ", accuracy_score(labels_test, prediction)

create_graph(clf, features_test, labels_test)
