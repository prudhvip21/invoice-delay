
import numpy as np
import pandas as pd
import os
from __future__ import division
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

os.chdir('/home/prudhvi/Documents')

# reading the data using inbuilt python parse function
data = pd.read_csv('WA_Accounts-Receivable.csv' , parse_dates = ['InvoiceDate','DueDate','SettledDate'])

#Seperating categorical
clean_data = data.iloc[:,[0,1,4,5,6,7,8,9,10,11]]

#Taking time stamp and extracting day,month and year
clean_data['Invoice_day'] = clean_data.apply(lambda x : x['InvoiceDate'].day,axis = 1)
clean_data['Invoice_month'] = clean_data.apply(lambda x : x['InvoiceDate'].month,axis = 1)
clean_data['Invoice_year'] = clean_data.apply(lambda x : x['InvoiceDate'].year,axis = 1)

#Deleting the dates after parsing
del clean_data['InvoiceDate']
del clean_data['DueDate']

# Parsing the settled day
clean_data['Settled_day'] = clean_data.apply(lambda x : x['SettledDate'].day,axis = 1)
clean_data['Settled_month'] = clean_data.apply(lambda x : x['SettledDate'].month,axis = 1)
clean_data['Settled_year'] = clean_data.apply(lambda x : x['SettledDate'].year,axis = 1)
#Deleting settledDate after parsiing
del clean_data['SettledDate']

""" Using label encoder for categorical data """

le = preprocessing.LabelEncoder()
clean_data['customerID_enc'] = le.fit_transform(clean_data['customerID'])
del clean_data['customerID']
clean_data['Disputed_enc'] = le.fit_transform(clean_data['Disputed'])
del clean_data['Disputed']

clean_data['PaperlessBill_enc'] = le.fit_transform(clean_data['PaperlessBill'])
del clean_data['PaperlessBill']

enc = OneHotEncoder()

encoded_data = enc.fit_transform(clean_data.iloc[:,[0,2,4,5,6,7,8,9,10,11,12]])


""" Feature names code """

cols = [0,2,4,5,6,7,8,9,10,11,12]
features = [ ]
for i in cols :
    cols_names = list(clean_data.columns)
    l = len(Counter(clean_data.iloc[:,i]))
    for k in range(l) :
        features.append(str(cols_names[i]) + '_' + str(k+1))




encoded_data = pd.DataFrame(encoded_data.toarray())

encoded_data['InvoiceAmount'] = clean_data['InvoiceAmount']


X_train, X_test, y_train, y_test = train_test_split(   encoded_data, clean_data.iloc[:,3], test_size=0.33, random_state=42)


regr1 =  DecisionTreeRegressor()

regr2 = AdaBoostRegressor(regr1, n_estimators=100)


regr.fit(X_train,y_train)

y_predict   = regr.predict(X_test)

print mean_squared_error(y_test, y_predict)

print regr.score(X_test,y_test)

parameters = [{ 'min_samples_split' : [0.1,0.15,0.2,0.3,2] , 'min_samples_leaf' : [1,5,10,15,20] , 'max_features': [10,30,50,80,100,150,200] , 'max_depth' :[10,20,30,40,50]}]

clf = GridSearchCV(regr1, param_grid= parameters, cv=5,
                   scoring= 'neg_mean_squared_error')

clf.fit(X_train,y_train)

print clf.best_params_

regr1 =  DecisionTreeRegressor(max_features=  50, min_samples_split= 2, max_depth = 40, min_samples_leaf = 1)

regr2 = AdaBoostRegressor(regr1)

parameters_ada = [{'n_estimators' : [10,20,30,50,80,100] ,'learning_rate' : [0.5,0.6,0.7,0.8,0.9,1], 'loss' : ['linear','square','exponential']}]

clf2 = GridSearchCV(regr2, param_grid= parameters_ada, cv=5, scoring= 'neg_mean_squared_error')

clf2.fit(X_train,y_train)

print clf2.best_params_

regr2 = AdaBoostRegressor(regr1,n_estimators = 100, loss= 'exponential' ,learning_rate =  0.7)

regr2.fit(X_train,y_train)

print regr2.score(X_test,y_test)


""" Ridge model """
ridge_regr = Ridge()
parameters_ridge = { 'alpha' : [0.25,0.5,0.75,1,1.25,2] , 'solver' : ['auto','svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'] }

ridge_regr = GridSearchCV(ridge_regr, param_grid= parameters_ridge, cv=5, scoring= 'neg_mean_squared_error')

ridge_regr.fit(X_train,y_train)

ridge_regr.best_params_

ridge_regr = Ridge(alpha=0.25 , solver= 'svd')
ada_ridge = AdaBoostRegressor(ridge_regr,n_estimators = 100, loss= 'exponential' ,learning_rate =  0.7)

ada_ridge.fit(X_train,y_train)

print ada_ridge.score(X_test,y_test)

""" Neural Network model """


nn_regr = MLPRegressor()

parameters_nn = { 'activation' : ['logistic', 'tanh', 'relu'],'alpha' : [0.001,0.005,0.01,0.025,0.05], 'learning_rate' : ['constant', 'invscaling', 'adaptive'] , 'momentum' : [0.25,0.5,0.75,1]}

nn_regr = GridSearchCV(nn_regr, param_grid= parameters_nn, cv=5, scoring= 'neg_mean_squared_error')



X = X_train.as_matrix().astype(np.float)
y = y_train.as_matrix().astype(np.float)

nn_regr.fit(X,y)

nn_regr.best_params_

nn_regr = MLPRegressor(activation = 'tanh',alpha = 0.001,
 learning_rate = 'adaptive',momentum = 0.5)

nn_adaboost = AdaBoostRegressor(nn_regr,n_estimators = 100, loss= 'exponential' ,learning_rate =  0.7)

nn_adaboost.fit(X,y)

print nn_adaboost.score(X_test,y_test)

