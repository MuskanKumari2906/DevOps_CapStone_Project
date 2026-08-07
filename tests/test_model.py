import os
import pandas as pd
from src.preprocess import preprocess_data, split_and_save
from src.train import train_and_evaluate
import joblib

def test_full_pipeline_on_dummy_data(tmp_path):
    # Create dummy data
    data = {
        "age": [50, 40, 60, 45, 55],
        "sex": [1, 0, 1, 0, 1],
        "cp": [1, 2, 3, 0, 1],
        "trestbps": [120, 130, 140, 110, 125],
        "chol": [200, 250, 300, 180, 220],
        "fbs": [0, 1, 0, 0, 0],
        "restecg": [0, 1, 2, 0, 1],
        "thalach": [150, 160, 140, 170, 155],
        "exang": [0, 1, 0, 0, 1],
        "oldpeak": [1.0, 2.0, 0.5, 0.0, 1.5],
        "slope": [1, 2, 0, 1, 2],
        "ca": [0, 1, 2, 0, 1],
        "thal": [2, 3, 2, 2, 3],
        "target": [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Preprocess
    X, y, scaler = preprocess_data(df)
    assert X.shape == (5, 13)
    assert len(y) == 5
    
    # Save
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    split_and_save(X, y, scaler, data_dir)
    
    assert (data_dir / "X_train.csv").exists()
    assert (data_dir / "scaler.joblib").exists()
    
    # Train and evaluate
    model_dir = tmp_path / "model"
    train_and_evaluate(str(data_dir), str(model_dir))
    
    assert (model_dir / "best_model.joblib").exists()
    
    # Load model and test predict
    model = joblib.load(model_dir / "best_model.joblib")
    preds = model.predict(X.iloc[:2])
    assert len(preds) == 2
