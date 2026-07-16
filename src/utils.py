import os 
import sys
#import numpy as np
import pandas as pd 
import dill
import pickle

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import customexception
from src.logger import logging



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models,params):
    from src.mlflow_tracker import start_experiment, log_model_run
    try:
        report = {}
        start_experiment()

        for model_name,model in models.items():
            para=params[model_name]
            rs = RandomizedSearchCV(
                    estimator=model,   # model to tune
                    param_distributions=para,  # params to try
                    n_iter=10,                   # how many random combinations to try
                    cv=3,                        # 5 fold cross validation
                    scoring='f1',                # metric to optimize
                    random_state=42              # reproducibility
                )
            rs.fit(X_train,y_train)
            model.set_params(**rs.best_params_)

            model.fit(X_train,y_train)


            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)

            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            accuracy = accuracy_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


            report[model_name] = test_model_score
            
            log_model_run(model_name, model, rs.best_params_, train_model_score, test_model_score,precision, recall, accuracy, roc_auc)

        return report
    
    except Exception as e:
        raise customexception(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customexception(e, sys)