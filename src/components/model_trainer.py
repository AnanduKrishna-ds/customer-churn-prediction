import os 
import sys

from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from src.exception import customexception
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig() 

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestClassifier(class_weight='balanced'),
                "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
                "Logistic Regression": LogisticRegression(class_weight='balanced'),
                "XGBClassifier": XGBClassifier(scale_pos_weight=3),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier()
            }

            params = {
                    "Random Forest": {
                        'n_estimators': [50, 100, 200, 300, 500],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "XGBClassifier": {
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [3, 5, 7, 9],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0]
                    },
                    "CatBoosting Classifier": {
                        'depth': [4, 6, 8, 10],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'iterations': [50, 100, 200, 300],
                        'class_weights': [[1, 3]]
                    },
                    "Logistic Regression": {
                        'C': [0.01, 0.1, 1, 10, 100],
                        'solver': ['lbfgs', 'liblinear', 'saga'],
                        'max_iter': [100, 200, 500]
                    },
                    "Decision Tree": {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    "AdaBoost Classifier": {
                        'learning_rate': [0.01, 0.1, 0.5, 1.0],
                        'n_estimators': [50, 100, 200, 300]
                    },
                    "KNeighborsClassifier": {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    }
                }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise customexception("No best Model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(X_test)

            F1_score=f1_score(y_test,predicted)

            return F1_score,best_model,model_report

        except Exception as e:
            raise customexception(e,sys)