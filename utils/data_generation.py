"""Create synthetic dataset = a CSV file"""
"""Label of a = last element in a raw. Label given as f."""
import datetime
import itertools
import os
import uuid
from itertools import combinations
from random import randint

import numpy as np

# FOR LOADING DATA
from pandas import read_csv
from sklearn.utils import shuffle


# clean an existing folder with a trailing / in the name
def clean_folder(folder):
    for f in os.listdir(folder):
        os.remove(folder + f)
    return


"""
    range is the values to be generated inclusive. 
    If _range is one, then the values are 0 or 1 (binary)
"""


def create_a_csv_row(f, dimension, _range=1):
    arr = [randint(0, _range) for _ in range(dimension)]
    row = ",".join(map(str, arr))
    label = f(arr)
    row += f",{label}"
    return row, arr


# size is the number of rows  - f is the class function - range of value (1 means binary)
def create_categorical_dataset(f, dimension, size, categorical_range):
    clean_folder("tests/")
    now = datetime.datetime.now()
    filename_out = f"tests/sample{now.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}.csv"

    # filename_out = f"tests/sample{size}-{categorical_range}-{now}.csv"
    set_arr = set()

    with open(filename_out, "w") as outfile:
        i = 0
        while i < size:
            row, arr = create_a_csv_row(f, dimension, categorical_range)
            if tuple(arr) not in set_arr:
                set_arr.add(tuple(arr))
                outfile.write(f"{row}\n")
                i += 1

    return filename_out


# GENERATE RANDOM SAMPLE SET OF size ELEMENTS FROM A DATASET
def generate_sample_set(dataset, size):
    rdataset = shuffle(dataset)
    sample_set = rdataset[:size]
    return sample_set


def pick_a_random_element(sample_set):
    n = randint(0, sample_set.shape[0] - 1)
    a = sample_set[n]
    return a


# COMPUTE ALL PAIRS FROM SAMPLE SET
def all_pairs(sample_set):
    set_of_pairs = list(combinations(sample_set, 2))
    return set_of_pairs


def load_dataset(filename):
    dataset = read_csv(filename, header=None)
    data = dataset.values
    X = data[:, :-1]
    y = data[:, -1]
    dimension = data.shape[1] - 1
    initial_size = data.shape[0]
    return data, X, y, dimension, initial_size


# FOR SOBOL
def create_dataAB(dataA, dataB, variable_index_to_fix):
    dataB_withA = dataB.copy()
    dataB_withA[:, variable_index_to_fix] = dataA[:, variable_index_to_fix]
    return dataB_withA
