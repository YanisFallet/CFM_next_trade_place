import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


def convert_feature_name(data : pd.DataFrame) -> pd.DataFrame:
    data.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) else col for col in data.columns]
    return data

def load_data(train=True, train_labels=True, test_data=False, remove_na=False) -> tuple:
    data, labels, test= None, None, None
    if train:
        data = pd.read_hdf("data/train_dc2020.h5", "data")
    if train_labels:
        labels = pd.read_csv("data/train_labels.csv")
    if test_data:
        test = pd.read_hdf("data/test_dc2020.h5", "data")
    
    results = [result for result in (data, labels, test) if result is not None]
    
    if remove_na:
        m = clean_data(data)
        results = [result[~m] for result in results]
    
    return tuple(results)

def compute_best_spread(data : pd.DataFrame) -> pd.Series:
    data_ = convert_feature_name(data)
    ask_columns = [f"ask_{i}" for i in range(6)]
    result = data_[ask_columns].min(axis=1)
    data_["spread"] = result
    return data_


def clean_data(data : pd.DataFrame) -> pd.DataFrame:
    return data.isna().any(axis=1)


def train_test_split_data(data : pd.DataFrame, labels : pd.DataFrame, test_size=0.2, random_state=None) -> tuple:
    return train_test_split(data, labels, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    data, labels = load_data(remove_na=True)
    print(data.shape, labels.shape)

    