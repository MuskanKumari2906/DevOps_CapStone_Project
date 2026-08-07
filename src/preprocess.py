import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler

def split_and_save(X, y, scaler, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)
    
    # Simple EDA
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    print(f"Saving processed data to {args.output_dir}...")
    split_and_save(X, y, scaler, args.output_dir)
    print("Preprocessing complete.")
