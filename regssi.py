from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,accuracy_score
import pandas as pd
import numpy as np

def train_test(df,input):
    x=df.drop(input,axis=1)
    y=df[input]
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=100)
    
    return xtrain,xtest,ytrain,ytest

def logr(xtrain,xtest,ytrain,ytest):
    model=LinearRegression().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    report=r2_score(ytest,ytest_preds)
    return model,report

def decisionr(xtrain,xtest,ytrain,ytest):
    model=DecisionTreeRegressor().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    report=r2_score(ytest,ytest_preds)
    return model,report

def randomr(xtrain,xtest,ytrain,ytest):
    model=RandomForestRegressor().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    report=r2_score(ytest,ytest_preds)
    return model,report

def gradientr(xtrain,xtest,ytrain,ytest):
    model=GradientBoostingRegressor().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    report=r2_score(ytest,ytest_preds)
    return model,report

def adar(xtrain,xtest,ytrain,ytest):
    model=AdaBoostRegressor().fit(xtrain,ytrain)
    ytest_preds=model.predict(xtest)
    ytrain_preds=model.predict(xtrain)
    report=r2_score(ytest,ytest_preds)
    return model,report