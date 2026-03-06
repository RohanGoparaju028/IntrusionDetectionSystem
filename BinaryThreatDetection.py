import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import sys

class BinaryThreat():
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
        self.le = LabelEncoder()
        self.sc = RobustScaler()

    def countZeroOne(self, df, col) -> ():
        zero = (df[col] == 0).sum()
        one = (df[col] == 1).sum()
        return (zero, one)

    def removeSparsity(self, df) -> pd.DataFrame:
        THRESHOLD = 0.99
        cols_to_drop = []
        for col in df.columns:
            counts = df[col].value_counts()
            if not counts.empty:
                most_freq = counts.iloc[0]
                if (most_freq / len(df)) > THRESHOLD:
                    cols_to_drop.append(col)
        df.drop(columns=cols_to_drop, inplace=True)
        return df

    def printColZeroOne(self, df, col) -> None:
        val = self.countZeroOne(df, col)
        print(f"{col} -> number of 0 {val[0]}, number of 1 {val[1]}")

    def labelEncoding(self, col: pd.Series) -> np.ndarray:
        return self.le.fit_transform(col.astype(str))

    def fit(self, x_train, y_train) -> None:
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    def modelevaluation(self, y_test, y_pred) -> None:
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Threat'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance (Check for Leakage)')
        plt.bar(range(10), importances[indices[:10]], align='center')
        plt.ylabel('Importance Score')
        plt.show()
        plt.savefig("Score.jpg")
    def visualize_results(self, df, X):
        df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Class Distribution (0: Normal, 1: Threat)')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.show()

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-10:]
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    def main(self, f: str) -> None:
        if not os.path.exists(f):
            sys.exit("The csv file does not exist")

        df = pd.read_csv(f)

        drop_cols = ["src_ip", "dst_ip", "type"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        df = df.replace("-", np.nan)
        threshold = 0.70
        limit = int(threshold * len(df))
        df.dropna(thresh=limit, axis=1, inplace=True)

        skewed_cols = ["src_bytes", "dst_bytes", "duration", "src_ip_bytes", "dst_ip_bytes", "missed_bytes"]
        for col in skewed_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = np.log1p(df[col].fillna(0))

        df = self.removeSparsity(df)

        for col in df.select_dtypes(include=["object"]).columns:
            if col != 'label':
                df[col] = self.labelEncoding(df[col])

        df.fillna(0, inplace=True)

        X = df.drop(columns=["label"])
        y = df["label"]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        x_train_scaled = self.sc.fit_transform(x_train)
        x_test_scaled = self.sc.transform(x_test)

        self.fit(x_train_scaled, y_train)
        y_pred = self.predict(x_test_scaled)

        self.modelevaluation(y_test, y_pred)
        self.visualize_results(df, X)

if __name__ == '__main__':
    f_path = "train_test_network.csv"
    obj = BinaryThreat()
    obj.main(f_path)
