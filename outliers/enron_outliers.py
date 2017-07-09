#!/usr/bin/python

import pickle
import matplotlib.pyplot
from feature_format import featureFormat


# read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]

# remove outlier
data_dict.pop("TOTAL", 0)

data = featureFormat(data_dict, features)


# your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
