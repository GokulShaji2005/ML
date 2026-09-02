import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix;
roc_curve,
auc
)

data= load_breast_cancer()
x=data.data
y=data.target
print("Dataset Shape:",x.shape)
print("Classes:",np.unique(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify =y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

mle_model= LogisticRegression(
    penalty=None,
    max_iter=5000,
    random_state=42
)
mle_model.fit(x_train,y_train)
y_pred_mle=mle_model.predict(x_test)

map_l2=LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=5000,
    random_state=42 
    )
map_l2.fit(x_train,y_train)
y_pred_l2=map_l2.predict(x_test)

map_l1=LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    max_iter=5000,
    random_state=42 
    )
map_l1.fit(x_train,y_train)
y_pred_l1=map_l1.predict(x_test)

def evaluate(model_name,y_true,y_pred):
    print("\n","="*40)
    print(model_name)
    print("="*40)

    print("Accuracy:",accuracy_score(y_true,y_pred))
    print("Precision:",precision_score(y_true,y_pred))
    print("Recall:",recall_score(y_true,y_pred))
    print("F1 Score:",f1score(y_true,y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true,y_pred))

evaluate("MLE",y_test,y_pred_mle)
evaluate("MLE(L2)",y_test,y_pred_l2)
evaluate("MLE(L1)",y_test,y_pred_l1)

print("Number of Zero Coefficients")
print("MLE :",np.sum(mle_model.coef_[0]==0))
print("MAP L2 :",np.sum(map_l2.coef_[0]==0 ))
print("MAP L1 :",np.sum(map_l1.coef_[0]==0 ))
