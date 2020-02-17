# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:12:07 2020

@author: WillW
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:27:38 2020
WhjWood 17/02/2020
A class for inductive conformal prediction for regression
based on Papadopoulos et al, (2011) 
@author: WillW
date of last edit: 17/02/2020 , 14:30
"""

# This first section sets up a toy model in order to run conformal prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

housing_data = pd.read_csv(r"C:\Users\willw\Documents\Datasets\House_Prices\train.csv", index_col="Id")

# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = list(housing_data.select_dtypes(include=[np.object]).columns)

# Select numerical columns
numeric_cols = list(housing_data.select_dtypes(include=[np.int64,np.float64]).columns)
numeric_cols.remove("SalePrice")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent", fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define model
model = SVR(kernel = "rbf", degree=4,  C=2000)

# Bundle preprocessing and modeling code in a pipeline
PL = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Remove rows with missing target, separate target from predictors
housing_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = housing_data.SalePrice
housing_data.drop(['SalePrice'], axis=1, inplace=True)


# Break off validation set from training data
X_train, X_split, y_train, y_split = train_test_split(housing_data, y, test_size= 0.4)
X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, test_size = 0.4)


# Keep selected columns only
my_cols = numeric_cols + categorical_cols


X_train = X_train[my_cols].copy()
X_val = X_val[my_cols].copy()


# Preprocessing of training data, fit model 
PL.fit(X_train, y_train)
yhat = PL.predict(X_val)

val_mae = mean_absolute_error(y_val ,yhat)
fraction_error = np.abs(y_val -yhat)/y_val 

print("Mean error (fraction)", np.mean(fraction_error))
print("Mean absolute error",val_mae)

#%% Conformal_Prediction class

class Conformal_Regression(object):
    
    def __init__(self):
        pass

    def fit(self,cal_X, cal_y, model=None, epsilon=None,gamma=None,rho=None):
        self.Underlying_Model = model
        self.Error_Rate = epsilon
        self.gamma = gamma
        self.rho = rho
        self.cal_y = cal_y
        if self.Underlying_Model == None:
            print("The underlying model has not been defined")
        if self.Error_Rate == None:
            print("The error rate (epsilon) has not been defined")
        from sklearn.neighbors import NearestNeighbors
        self.nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(cal_X)
        

        alpha = self.Nonconformity(cal_X,cal_y, yhat)
        alpha_list = alpha[0,:].tolist()
        
        alpha_list.sort()
        #self.a_error = np.array(alpha_list)[[np.cumsum(alpha_list / np.sum(alpha_list)) >= 1 - self.Error_Rate]].min()
        self.a_error = min(alpha_list[int(len(alpha_list)*(1-self.Error_Rate)) :] )

    
    def Nonconformity(self,X,y, yhat):

        distances, indices = self.nbrs.kneighbors(X)
        self.median_distance = np.mean(distances) #change back to median
        d = np.sum(distances, axis=1) / self.median_distance
        
        S = np.std(y[indices],axis=1)
        self.median_S = np.median(S)
        S = S / self.median_S 
        yhat = self.Underlying_Model.predict(X)
            
        alpha = np.abs(y-yhat.reshape(1,-1)) / (np.exp(self.gamma*d) + np.exp(self.rho*S) )
        return alpha
    
    def predict(self, X):
        yhat = self.Underlying_Model.predict(X)
        distances, indices = self.nbrs.kneighbors(X)

        d = np.sum(distances, axis=1) / self.median_distance
        
        S = np.std(self.cal_y[indices],axis=1)
        S = S / self.median_S
        norm = np.exp(self.gamma*d) + np.exp(self.rho*S) 
        
        error = self.a_error*norm
        return yhat.reshape(1,-1), error.reshape(1,-1)
#%% Use of conformal predictor     
cal_X = preprocessor.transform(X_val).toarray()
cal_y = y_val.values

test_X = preprocessor.transform(X_test).toarray()
test_y = y_test.values
Model = PL["model"]


CP = Conformal_Regression()
CP.fit(cal_X, cal_y, Model, epsilon=0.1,gamma=0.1,rho=0.1)

yhat, error = CP.predict(test_X)
#plt.scatter(test_y.reshape(1,-1), yhat.reshape(1,-1))
plt.axis([0,1.2*np.max(test_y),0,1.2*np.max(yhat)])

plt.errorbar(test_y.reshape(1,-1)[0,:], yhat.reshape(1,-1)[0,:], yerr=error.reshape(1,-1)[0,:],fmt='o', color='black',
             ecolor='lightgray')
plt.show()


