#!/usr/bin/python

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    # your code goes here
    remove_count = int(len(predictions) * 0.1)
    errors = np.abs(net_worths - predictions).reshape((1, len(predictions)))
    sorted_index = np.argsort(errors)

    for i in range(len(predictions) - remove_count):
        idx = sorted_index[0][i]
        tpl = (ages[idx], net_worths[idx], errors[0][idx])
        cleaned_data.append(tpl)
    
    return cleaned_data

