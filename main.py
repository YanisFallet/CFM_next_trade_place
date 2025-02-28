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
from sklearn.model_selection import GridSearchCV

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
    
    
def XGB_grid_search():
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    model = XGBClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    for params in tqdm(list(ParameterGrid(param_grid))):
        model.set_params(**params)
        model.fit(train_data, train_labels["source_id"])
        
        test_pred = model.predict(test_data)
        accuracy = accuracy_score(test_labels["source_id"], test_pred)
        
        # Save the model with accuracy score and parameters in the filename
        model_filename = f"models/XGB_{accuracy:.4f}_params_{params}.model"
        model.save_model(model_filename)
        
        print(f"Saved model: {model_filename} with accuracy: {accuracy}")
    
def LR():
    model = LogisticRegressionCV(max_iter=1000, verbose=5)
    model.fit(train_data, train_labels["source_id"])
    test_pred = model.predict(test_data)
    print(accuracy_score(test_labels["source_id"], test_pred))
    print(confusion_matrix(test_labels["source_id"], test_pred))
    
    
if __name__ == "__main__":
