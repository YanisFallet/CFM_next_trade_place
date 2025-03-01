import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

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
from sklearn.model_selection import ParameterGrid

import optuna
from optuna.samplers import TPESampler
import os

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
    
    with open("models_xgb/score.json", "r") as f:
        scores = json.load(f)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }
    
    for params in tqdm(list(ParameterGrid(param_grid))):
        if str(params) in scores:
            print(f"Skipping {params}")
            continue
        model = XGBClassifier()
        model.set_params(**params)
        model.fit(train_data, train_labels["source_id"])
        
        test_pred = model.predict(test_data)
        accuracy = accuracy_score(test_labels["source_id"], test_pred)
        
        with open("models_xgb/score.json", "w") as f:
            scores[str(params)] = accuracy
            json.dump(scores, f, indent=4)
        
        # Save the model with accuracy score and parameters in the filename
        model_filename = f"models_xgb/XGB_{accuracy:.4f}_params_{params}.model"
        model.save_model(model_filename)
        
        print(f"Saved model: {model_filename} with accuracy: {accuracy}")
    

def XGB_optuna_search(n_trials=100):
    # Créer le dossier s'il n'existe pas
    os.makedirs("models_xgb", exist_ok=True)
    
    # Charger les scores existants ou créer un nouveau dictionnaire
    try:
        with open("models_xgb/score.json", "r") as f:
            scores = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        scores = {}
    
    def objective(trial):
        # Définir l'espace de recherche des hyperparamètres
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 3.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        
        # Vérifier si cette combinaison de paramètres a déjà été évaluée
        params_str = str(params)
        if params_str in scores:
            return scores[params_str]
        
        # Entraîner le modèle
        model = XGBClassifier(**params)
        model.fit(train_data, train_labels["source_id"])
        
        # Évaluer le modèle
        test_pred = model.predict(test_data)
        accuracy = accuracy_score(test_labels["source_id"], test_pred)
        
        # Sauvegarder le score
        scores[params_str] = accuracy
        with open("models_xgb/score.json", "w") as f:
            json.dump(scores, f, indent=4)
        
        # Sauvegarder le modèle si c'est le meilleur jusqu'à présent
        trial_value = trial.number if hasattr(trial, 'number') else len(scores)
        model_filename = f"models_xgb/XGB_{accuracy:.4f}_trial_{trial_value}.model"
        model.save_model(model_filename)
        
        print(f"Trial {trial_value}: accuracy={accuracy:.4f} with params={params}")
        
        return accuracy

    # Utiliser le sampler TPE pour une recherche bayésienne efficace
    sampler = TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    # Afficher les meilleurs résultats
    print("Meilleurs hyperparamètres trouvés :")
    print(study.best_params)
    print(f"Meilleure précision : {study.best_value:.4f}")

    # Sauvegarder le meilleur modèle avec un nom spécial
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(train_data, train_labels["source_id"])
    best_model.save_model(f"models_xgb/XGB_best_{study.best_value:.4f}.model")
    
    return study.best_params, study.best_value


def LR():
    model = LogisticRegressionCV(max_iter=1000, verbose=5)
    model.fit(train_data, train_labels["source_id"])
    test_pred = model.predict(test_data)
    print(accuracy_score(test_labels["source_id"], test_pred))
    print(confusion_matrix(test_labels["source_id"], test_pred))
    
    
if __name__ == "__main__":
    XGB_optuna_search(n_trials=100)