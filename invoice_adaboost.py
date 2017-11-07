
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

os.chdir('/home/prudhvi/Documents')

data = pd.read_csv('WA_Accounts-Receivable.csv' , parse_dates = ['InvoiceDate','DueDate','SettledDate'])

data.head(10)

clean_data = data.iloc[:,[0,1,4,5,6,7,8,9,10,11]]

clean_data['Invoice_day'] = clean_data.apply(lambda x : x['InvoiceDate'].day,axis = 1)

clean_data['Invoice_month'] = clean_data.apply(lambda x : x['InvoiceDate'].month,axis = 1)

clean_data['Invoice_year'] = clean_data.apply(lambda x : x['InvoiceDate'].year,axis = 1)

del clean_data['InvoiceDate']
del clean_data['DueDate']


clean_data['Settled_day'] = clean_data.apply(lambda x : x['SettledDate'].day,axis = 1)

clean_data['Settled_month'] = clean_data.apply(lambda x : x['SettledDate'].month,axis = 1)

clean_data['Settled_year'] = clean_data.apply(lambda x : x['SettledDate'].year,axis = 1)


del clean_data['SettledDate']


clean_data.head(5)


le = preprocessing.LabelEncoder()
clean_data['customerID_enc'] = le.fit_transform(clean_data['customerID'])
del clean_data['customerID']
clean_data['Disputed_enc'] = le.fit_transform(clean_data['Disputed'])
del clean_data['Disputed']

clean_data['PaperlessBill_enc'] = le.fit_transform(clean_data['PaperlessBill'])
del clean_data['PaperlessBill']

enc = OneHotEncoder()

encoded_data = enc.fit_transform(clean_data.iloc[:,[0,2,4,5,6,7,8,9,10,11,12]])

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


regr2 = AdaBoostRegressor(regr1,n_estimators = 100, loss= 'exponential' ,learning_rate =  0.7)

regr2.fit(X_train,y_train)

print regr2.score(X_test,y_test)