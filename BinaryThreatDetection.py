import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
import os
import sys
class BinaryThreat():
    def __init__(self):
        self.model = LogisticRegression(max_iter=10000,class_weight="balanced")
        self.le = LabelEncoder()
        self.sc = RobustScaler()
    def countZeroOne(self,df:DataFrame,col:str) -> ():
        zero = (df[col] == 0).sum()
        one = (df[col] == 1).sum()
        return (zero,one)
    def removeSparsity(self,df:DataFrame) -> DataFrame:
        THRESHOLD = 0.99
        for col in df.columns:
            most_freq = df[col].value_counts().iloc[0]
            if (most_freq / len(df[col])) > THRESHOLD:
                df.drop(columns=[col],inplace = True)
        return df
    def printColZeroOne(self,df:DataFrame,col:str) -> None:
        val = self.countZeroOne(df,col)
        print(f"{col} -> number of 0 {val[0]},number of 1 {val[1]}")
    def labelEncoding(self,col:pd.Series) -> np.ndarray:
        return self.le.fit_transform(col.astype(str))
    def fit(self,x_train,y_train) -> None:
        self.model.fit(x_train,y_train)
    def predict(self,x_test):
        return self.model.predict(x_test)
    def modelevaluation(self,y_test,y_pred) -> None:
        acc = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        print(f"Accuracy:{acc:.4}")
        print(f"precision:{precision:.4}")
        print(f"recall:{recall:.4}")
        print(f"f1_score:{f1:.4}")
    def main(self,f:str) -> None :
        if not os.path.exists(f):
            sys.exit("The csv file does not exists")
        df = pd.read_csv(f)
        df.drop(columns=["src_ip","dst_ip","type"],inplace = True)
        df = df.replace("-",np.nan)
        threshold = 0.70
        limit = threshold * len(df)
        print(df.isnull().sum())
        df.dropna(thresh = limit,axis=1,inplace = True)
        print(f"After reducing the dimensionality reduction\n{df.isnull().sum()}")
        for col in df.columns:
            self.printColZeroOne(df,col)
        skewed_cols = ["src_bytes", "dst_bytes", "duration", "src_ip_bytes", "dst_ip_bytes", "missed_bytes"]
        for col in skewed_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].astype(float))
        print("Removing the most frequent elements from the columns that causes sparsity")
        df = self.removeSparsity(df)
        for col in df.columns:
            self.printColZeroOne(df,col)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = self.labelEncoding(df[col])
        df.fillna(0,inplace=True)
        X = df.drop(columns=["label"])
        y = df["label"]
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        x_train = self.sc.fit_transform(x_train)
        x_test = self.sc.transform(x_test)
        self.fit(x_train,y_train)
        y_pred = self.predict(x_test)
        self.modelevaluation(y_test,y_pred)
if __name__ == '__main__':
    f:str = "train_test_network.csv"
    obj = BinaryThreat()
    obj.main(f)
