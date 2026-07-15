import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost


def start_experiment(experiment_name="Customer_Churn_Prediction"):
    """Call this once, before the training loop starts."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)


def log_model_run(model_name, model, best_params, train_score, test_score,precision, recall, accuracy, roc_auc):
    """Logs one classifier's run: its best params, train/test F1, and the model itself."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(best_params)
        mlflow.log_metric("train_f1_score", train_score)
        mlflow.log_metric("test_f1_score", test_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # Use the correct flavor based on model type
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, name="model")
        elif "CatBoost" in model_name:
            mlflow.catboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")