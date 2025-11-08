import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.preprocessing import preprocess_data

def preprocess_data(data):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    target = 'alert'

    X = df[features]
    y = df[target]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded

