import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json

def train_model(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    features = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
    target = 'alert'
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_regression_model.pkl")
    metrics = {"accuracy": acc}
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    train_model()
