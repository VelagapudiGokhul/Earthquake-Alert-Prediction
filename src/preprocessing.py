import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(data=None, input_path="data/raw/earthquake_dataset.csv", output_dir="data/processed/"):
    if data is not None:
        df = data.copy()
    else:
        df = pd.read_csv(input_path)

    df = df.dropna().drop_duplicates()
    features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    target = 'alert'
    X = df[features]
    y = df[target]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if data is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        train_df = pd.DataFrame(X_train, columns=features)
        train_df["alert"] = y_train
        test_df = pd.DataFrame(X_test, columns=features)
        test_df["alert"] = y_test
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoder, "models/label_encoder.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

    return X_scaled, y_encoded

if __name__ == "__main__":
    preprocess_data()
