import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    
    df = pd.read_csv('data/raw/earthquake_dataset.csv')
    X = scaler.transform(df[['magnitude','depth','cdi','mmi','sig']])
    y_true = encoder.transform(df['alert'])
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    assert acc >= 0.6
