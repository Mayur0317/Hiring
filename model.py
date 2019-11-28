# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:40:14 2019

@author: Dell
"""
# Importing File
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('C:\\Users\\Dell\\Desktop\\DataScience\\My_Work\\hiring\\Deployment-flask-master\\Deployment-flask-master\\hiring.csv')

dataset['experience'].fillna(0,inplace = True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace = True)
X = dataset.iloc[:, :3]

# Convert Text to Integer
def convert_to_integer(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
X['experience'] = X['experience'].apply(lambda x: convert_to_integer(x))

y = dataset.iloc[:, -1]

#print(dataset)
print(X)
print(y)

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#fitting model with trainning data
regressor.fit(X,y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
