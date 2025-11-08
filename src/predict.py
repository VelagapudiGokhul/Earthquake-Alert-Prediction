import pandas as pd
import joblib

def predict_alert(new_data):
    model = joblib.load("models/logistic_regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    if isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    else:
        df = pd.DataFrame(new_data, columns=['magnitude', 'depth', 'cdi', 'mmi', 'sig'])
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    alert_labels = encoder.inverse_transform(preds)
    return alert_labels

if __name__ == "__main__":
    sample = {"magnitude": 5.8, "depth": 12.0, "cdi": 3.2, "mmi": 4.0, "sig": 350}
    result = predict_alert(sample)
    print("Predicted Alert:", result[0])
