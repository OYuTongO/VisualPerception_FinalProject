import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_CSV   = os.path.join("data", "landmarks.csv")
MODEL_PATH = os.path.join("model", "asl_classifier.pkl")

def load_data():
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("训练 RandomForestClassifier ...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  RandomForest 测试集准确率: {acc:.4f}")

    if acc < 0.90:
        print("准确率未达90%，切换至 MLPClassifier ...")
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  MLP 测试集准确率: {acc:.4f}")

    return clf, acc

def main():
    print("读取数据 ...")
    X, y = load_data()
    print(f"  样本数: {len(X)}  类别数: {len(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  训练集: {len(X_train)}  测试集: {len(X_test)}\n")

    clf, acc = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n各字母详细报告:")
    print(classification_report(y_test, clf.predict(X_test), digits=3))

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"模型已保存至 {MODEL_PATH}")
    print(f"\n最终准确率: {acc:.2%}")

if __name__ == "__main__":
    main()
