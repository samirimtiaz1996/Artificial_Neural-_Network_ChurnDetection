# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:10:04 2020

@author: Samir Imtiaz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset=pd.read_csv('Churn_Modelling.csv')
T=dataset.iloc[:,3:4].values
t=dataset.iloc[:,3].values
print(T)
print(t)
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

labelencoder_X_country= LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])


labelencoder_X_gender= LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])


# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
# import keras
# from  keras.models import Sequential
# from keras.layers import Dense
# 
# classifier = Sequential()
# 
# 
# classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
# 
# classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
# 
# classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# 
# classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# 
# classifier.fit(X_train,y_train,batch_size=10,epochs=100)
# 
# y_pred=classifier.predict(X_test)
# y_pred=(y_pred>.5)
# 
# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,y_pred)
# print(cm)
# 
# =============================================================================
# With k-fold cross validation

import keras
from  keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# =============================================================================
# def build_classifier(optimizer):
#   classifier = Sequential()
#   classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
#   classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
#   classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#   classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
#   return classifier
# 
# classifier=KerasClassifier(build_fn=build_classifier)
# parameters={'batch_size':[25,32],
#             'nb_epoch':[100,500],
#             'optimizer':['adam','rmsprop']}
# 
# grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
# grid_search=grid_search.fit(X_train,y_train)
# best_parameters=grid_search.best_params_
# best_accuracy=grid_search.best_score_
# 
# 
# =============================================================================
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
  classifier.add(Dropout(.1))
  classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu')) 
  classifier.add(Dropout(.1))
  classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
  classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=1,epochs=500)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=1)
print(accuracies.mean())
print(accuracies.std())
# =============================================================================
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn = build_classifier)
# parameters = {'batch_size': [25, 32],
#               'epochs': [100, 500],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train, y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
# print(best_accuracy)
# =============================================================================
3.59
