from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
import pandas as pd
import numpy as np

def train_test(df,input):
    x=df.drop(input,axis=1)
    y=df[input]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=100)
    
    return xtrain,xtest,ytrain,ytest

def log(xtrain,xtest,ytrain,ytest):
    model=LogisticRegression().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    test_acc=accuracy_score(ytest,ytest_preds)
    train_acc=accuracy_score(ytrain,ytrain_preds)
    report=classification_report(ytest,ytest_preds,output_dict=True)
    return model,test_acc,train_acc,report

def decision(xtrain,xtest,ytrain,ytest):
    model=DecisionTreeClassifier().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    test_acc=accuracy_score(ytest,ytest_preds)
    train_acc=accuracy_score(ytrain,ytrain_preds)
    report=classification_report(ytest,ytest_preds,output_dict=True)
    return model,test_acc,train_acc,report

def random(xtrain,xtest,ytrain,ytest):
    model=RandomForestClassifier().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    test_acc=accuracy_score(ytest,ytest_preds)
    train_acc=accuracy_score(ytrain,ytrain_preds)
    report=classification_report(ytest,ytest_preds,output_dict=True)
    return model,test_acc,train_acc,report

def gradient(xtrain,xtest,ytrain,ytest):
    model=GradientBoostingClassifier().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    test_acc=accuracy_score(ytest,ytest_preds)
    train_acc=accuracy_score(ytrain,ytrain_preds)
    report=classification_report(ytest,ytest_preds,output_dict=True)
    return model,test_acc,train_acc,report

def ada(xtrain,xtest,ytrain,ytest):
    model=AdaBoostClassifier().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    test_acc=accuracy_score(ytest,ytest_preds)
    train_acc=accuracy_score(ytrain,ytrain_preds)
    report=classification_report(ytest,ytest_preds,output_dict=True)
    return model,test_acc,train_acc,report






    

