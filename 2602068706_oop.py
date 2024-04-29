# -*- coding: utf-8 -*-
"""2602068706_OOP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JiMLrV1diQ_5cpxs_0OovlAhDt5NJx7v

#### DTSC6012001 - Model Deployment
Nama: Vira Fitriyani<br>
NIM: 2602068706<br>
MID Exam<br>
"""

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class XGB_model:
  def __init__(self):
    self.model = XGBClassifier()

  def train(self, x_train, y_train):
    self.model.fit(x_train, y_train)

  def evaluate(self, x_test, y_test):
    y_predict = self.model.predict(x_test)
    evaluation = classification_report(y_test, y_predict)
    return evaluation

  def predict(self, x):
    return self.model.predict(x)

path = 'XGB_model.pkl'