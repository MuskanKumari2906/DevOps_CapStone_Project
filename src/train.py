import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os
import argparse

def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_prob):
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5 # Default if only one class
        
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc
    }

def train_and_evaluate(data_dir, model_dir):
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = -1
    best_model_name = ""
    
    if os.getenv("GITHUB_ACTIONS") or os.getenv("CI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    else:
        mlflow.set_tracking_uri("https://dagshub.com/muskankumariv29/MLflow.mlflow")
        
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            metrics = evaluate_model(y_test, y_pred, y_prob)
            
            mlflow.log_params({"model_type": name})
            mlflow.log_metrics(metrics)
            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as e:
                print(f"Skipping MLflow model logging for {name} due to: {e}")
            
            print(f"Model: {name} | Metrics: {metrics}")
            
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_model_name = name
                
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the best model")
    args = parser.parse_args()
    
    train_and_evaluate(args.data_dir, args.model_dir)
