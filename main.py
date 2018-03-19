import pandas as pd 
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def give_headers(dataset):
    return list(dataset.columns.values)
    
def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model
    
def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

train_titanic = pd.read_csv("traintitanic-pp1.txt")
train_titanic = train_titanic.fillna(method='ffill')
print ("Number of samples :", len(train_titanic))
print (train_titanic.head())

headers = give_headers(train_titanic)
print("Data set headers : ", headers)

training_features = ['Pclass' , 'Sex' , 'SibSp' , 'Parch' , 'Fare' , 'Embarked']
traget = 'Survived'

train_x, test_x, train_y, test_y = train_test_split(train_titanic[training_features], train_titanic[traget], train_size=0.9)

print ("train_x size : ", train_x.shape)
print ("train_y size : ", train_y.shape)

print(train_x.head())
print(train_y.head())
 
print ("test_x size : ", test_x.shape)
print ("test_y size : ", test_y.shape)

trained_logistic_regression_model = train_logistic_regression(train_x, train_y)

train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)
print("Train Accuracy : ", train_accuracy)

test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y) 
print("Test Accuracy : ", test_accuracy)
