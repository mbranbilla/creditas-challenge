import pandas as pd
import pickle
import numpy as np

def load_dataset(dataset, dtypes, path="../data/"):
    return pd.read_csv(path + dataset, dtype=dtypes)


def save_pickle(object_to_save, filename, path="../mode_files/"):
    pd.to_pickle(object_to_save, path + filename)


def load_pickle(filename, path="../model_files/"):
    return pickle.Unpickler(open(path + filename, "rb"))


def filter_outliers(dataframe, column, n_stds=3):
    dataframe_filtered = dataframe[
        np.abs(dataframe[column] - dataframe[column].mean()) <=
        (n_stds * dataframe[column].std())
    ]
    return dataframe_filtered
