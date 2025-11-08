import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.preprocessing import preprocess_data

def test_preprocessing_shapes():
    df = pd.DataFrame({
        'magnitude':[4.5, 5.2],
        'depth':[10, 30],
        'cdi':[3.5, 4.0],
        'mmi':[2.0, 3.0],
        'sig':[150, 300],
        'alert':['green','red']
    })
    X_scaled, y_encoded = preprocess_data(df)
    assert X_scaled.shape == (2,5)
    assert len(y_encoded) == 2
