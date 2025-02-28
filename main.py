import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier

from xgboost import XGBClassifier
from xgboost import plot_importance

import utils

data, labels = utils.load_data(convert_features_n= True)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

def RF(n_estimators = 5):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, verbose=4)
    model.fit(train_data, train_labels["source_id"])
    test_pred = model.predict(test_data)
    print(accuracy_score(test_labels["source_id"], test_pred))
    print(confusion_matrix(test_labels["source_id"], test_pred))


def XGB(n_estimators = 100):
    model = XGBClassifier(n_estimators = n_estimators)
    model.fit(train_data, train_labels["source_id"], verbose = 5)
    output = model.predict(test_data)
    print(accuracy_score(output, test_labels["source_id"]))
    print(confusion_matrix(output, test_labels["source_id"]))
    # model.save_model("models/XGB.pth")
    # plot_importance(model)
    # plt.show()
    
    
if __name__ == "__main__":
    XGB()