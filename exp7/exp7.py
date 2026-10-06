import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score)
from sklearn.svm import SVC

print("Downloading Fashion MNIST dataset");

X,y=fetch_openml('Fashion-MNIST',version=1,return_X_y=True,as_frame=False)
X=X.astype('float32')
y=y.astype('int')

print("Dataset Shape",X.shape)
X=X[:10000]
y=y[:10000]

x_train,x_test,y_train,y_test=train_test_split(X,y,
    test_size=0.2,random_state=42,stratify=y
)

print("Training Sample",x_train.shape[0])
print("Testing Sample",x_test.shape[0])

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print("nTraining Linear SVM...")
linear_svm=SVC(kernel='linear')
linear_svm.fit(x_train,y_train)
