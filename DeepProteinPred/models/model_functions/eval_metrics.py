"""
Functions for evaluating the trained model.
"""


from sklearn.metrics import mean_squared_error
from math import sqrt
import random


def sample_dict(dict1, dict2, num):
    """
    Randomly sample a subset of two dictionaries 
    with the same keys
    """

    sub_dict1 = dict()
    sub_dict2 = dict()
    keys = random.sample(list(dict1), num)

    for key in keys:
        sub_dict1[key] = dict1[key]
        sub_dict2[key] = dict2[key]

    return sub_dict1, sub_dict2


def calc_rmse(cmap_preds, cmap_actual):
    """
    Given a dictionary of amino acid 1 hot encodings,
    calculate the average root mean squared error.
    """

    rmses = []
    for pdb_id, cmap in cmap_actual.items():
        cmap_pred = cmap_preds[pdb_id]

        rmse = sqrt(mean_squared_error(cmap_pred, cmap))
        rmses.append(rmse)

    avg = sum(rmses) / len(rmses)
    return avg