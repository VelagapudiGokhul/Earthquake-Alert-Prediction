import pandas as pd

def test_no_missing_values():
    df = pd.read_csv('data/raw/earthquake_dataset.csv')
    assert df.isnull().sum().sum() == 0

def test_columns_exist():
    df = pd.read_csv('data/raw/earthquake_dataset.csv')
    required_cols = ['magnitude','depth','cdi','mmi','sig','alert']
    for col in required_cols:
        assert col in df.columns
