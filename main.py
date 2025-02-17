import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier

import utils


data = pd.read_hdf("data/train_dc2020.h5", "data")
labels = pd.read_csv("data/train_labels.csv")

data = utils.convert_feature_name(data)
print(data.columns)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

print(train_data.head(), train_labels["source_id"].values)


model = RandomForestClassifier(n_estimators=10, random_state=42, verbose=4)
model.fit(train_data, train_labels["source_id"])
test_pred = model.predict(test_data)
print(accuracy_score(test_labels["source_id"], test_pred))
print(confusion_matrix(test_labels["source_id"], test_pred))