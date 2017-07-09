#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


# load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# the input features we want to use
# can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

# Mini project
stock_option_list = np.array(finance_features)[:, 1]
print "max of exercised_stock_options : ", np.max(stock_option_list)
print "min of exercised_stock_options : ", np.min(stock_option_list[stock_option_list != 0.0])
salary_list = np.array(finance_features)[:, 0]
print "max of salary : ", np.max(salary_list)
print "min of salary : ", np.min(salary_list[salary_list != 0.0])


# in the "clustering with 3 features" part of the mini-project,
# you'll want to change this line to
# for f1, f2, _ in finance_features:
# (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

# cluster here; create predictions of the cluster labels
# for the data and store them to a list called pred
cluster = KMeans(n_clusters=2)
cluster.fit(finance_features)
predictions = cluster.labels_

# rename the "name" parameter when you change the number of features
# so that the figure gets saved to a different file
try:
    Draw(predictions, finance_features, poi, mark_poi=False, name="clusters_by3feature.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
