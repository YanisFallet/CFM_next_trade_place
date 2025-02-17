import pandas as pd
import numpy as np

def convert_feature_name(data : pd.DataFrame) -> pd.DataFrame:
    data.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) else col for col in data.columns]
    return data

def load_data(train=True, train_labels=True, test_data=False):
    data, labels, test= None, None, None
    if train:
        data = pd.read_hdf("data/train_dc2020.h5", "data")
    if train_labels:
        labels = pd.read_csv("data/train_labels.csv")
    if test_data:
        test = pd.read_hdf("data/test_dc2020.h5", "data")
    
    results = [result for result in (data, labels, test) if result is not None]
    return tuple(results)

def compute_best_spread(data : pd.DataFrame) -> pd.Series:
    data_ = convert_feature_name(data)
    ask_columns = [f"ask_{i}" for i in range(6)]
    result = data_[ask_columns].min(axis=1)
    data_["spread"] = result
    return data_


if __name__ == "__main__":
    data, labels = load_data()
    print(compute_best_spread(data))

    