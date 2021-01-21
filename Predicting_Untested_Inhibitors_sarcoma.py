# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:37:33 2019

@author: svijay
"""



# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import mean_squared_error
from statistics import mean

response_data2 = pd.read_csv('Huh7_WT_Fzd2.csv')
drug_list2 = response_data2.iloc[:, 0].values
alldrugs2 = pd.read_csv('kir_allDrugs_namesDoses.csv', encoding='latin1')

alldrugs2 = alldrugs2.set_index('compound')
dataset2 = alldrugs2.loc[drug_list2]
response2 = response_data2['Huh7_Fzd2'].values
dataset2["response"] = response2


kinase_list = pd.read_csv('recursive_elimination_kinases_Huh7_Both.csv')
kinase_list = kinase_list.values.tolist()

kinases = []
for kinase in kinase_list:
    kinases.append(kinase[0])
    
# Importing the dataset
X = dataset2[kinases].values
y = dataset2.iloc[:, 298].values


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu', input_dim = len(kinases)))
classifier.add(Dense(units = 100, kernel_initializer = 'TruncatedNormal', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'TruncatedNormal' )) 
classifier.compile(loss = 'mean_squared_error', optimizer='adam')
classifier.fit(X,y, epochs=80, batch_size=44)

X_predict = alldrugs2
X_predict = X_predict[kinases]
#X_predict = alldrugs.loc[alldrugs.index.difference(drug_list)]
prediction_index = X_predict.index.tolist()
X_predict = X_predict.iloc[:, 0:len(kinases)+2].values

# Predicting the Test set results
y_pred = classifier.predict(X_predict)

untested_inhibitor_prediction = pd.DataFrame(y_pred.tolist(), index = prediction_index)




ranked_inhibitors = untested_inhibitor_prediction.sort_values(by=[0])


ranked_inhibitors.to_excel("Ranked_inhibitors Hs578t Best Model.xlsx", sheet_name='1')



