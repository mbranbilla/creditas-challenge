import pandas as pd
import pickle

def load_dataset(dataset, dtypes, path="../data/"):
    return pd.read_csv(path + dataset, dtype=dtypes)


def save_pickle(object_to_save, filename, path="../mode_files/"):
    pickle.dump(object_to_save, open(path + filename, "wb"))


def load_pickle(filename, path="../model_files/"):
    return pickle.load(open(path + filename, "wb"))

