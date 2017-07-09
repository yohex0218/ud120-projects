#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# no. of person
print type(enron_data) # dict
print len(enron_data.keys()) #146

# no. of attribute by person
data_by_person = enron_data["METTS MARK"]
print type(data_by_person) #dict
print data_by_person.keys()
print len(data_by_person.keys())#21

# no. of person who is poi
poi_count = 0
for key, value in enron_data.items():
    if value["poi"]:
        poi_count += 1
print poi_count #18

# What is the total value of the stock belonging to James Prentice?
print enron_data["PRENTICE JAMES"]["total_stock_value"]

# How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# What's the value of stock options exercised by Jeffrey K Skilling ?
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# Of these three individuals (Lay, Skilling and Fastow), 
# who took home the most money (largest value of "total_payments" feature)?
print "LAY KENNETH L", enron_data["LAY KENNETH L"]["total_payments"]
print "SKILLING JEFFREY K", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "FASTOW ANDREW S", enron_data["FASTOW ANDREW S"]["total_payments"]

# How many folks in this dataset have a quantified salary? What about a known email address?
have_salary = 0
have_email = 0
for key, value in enron_data.items():
    salary = value["salary"]
    email = value["email_address"]
    if salary != "NaN":
        have_salary += 1
    if email != "NaN":
        have_email += 1

print have_salary, have_email

# How many people in the E+F dataset (as it currently exists) have "NaN" for their total payments? 
# What percentage of people in the dataset as a whole is this?
have_nan_payments = 0
for key, value in enron_data.items():
    payments = value["total_payments"]
    if payments == "NaN":
        have_nan_payments += 1

print have_nan_payments

# How many POIs in the E+F dataset have "NaN" for their total payments? 
# What percentage of POI's as a whole is this?
have_nan_payments_poi = 0
for key, value in enron_data.items():
    poi = value["poi"]
    payments = value["total_payments"]
    if poi and payments == "NaN":
        have_nan_payments_poi += 1

print float(have_nan_payments_poi) / float(poi_count)
