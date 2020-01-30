import numpy as np
import pandas as pd
import pickle
import csv

def load_data():
    training_data = pd.read_csv('mnist_train.csv', dtype='uint8')
    test_data = pd.read_csv('mnist_test.csv', dtype='uint8')
    training_data = pd.DataFrame(training_data).to_numpy()
    test_data = pd.DataFrame(test_data).to_numpy()
    return (training_data, test_data)


def load_data_wrapper():
    tr_d, te_d = load_data()
    return (tr_d, te_d)