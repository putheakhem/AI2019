import numpy as np
import pandas as pd
import csv 
from sklearn import preprocessing
import warnings 
warnings.filterwarnings("ignore")

trainset = pd.read_csv("IMDBMovieData.csv", encoding='latin-1')

X = trainset.drop(['Title', 'ID', 'Votes', 'Year', 'Revenue', 'Metascore', 'Rating', 'Description', 'Runtime'], axis=1)

features = ['Genre', 'Actors', 'Director']

for feature in features:
    X_dummy = X[feature].str.get_dummies(',').add_prefix(feature + '.')

    X = X.drop([feature], axis =1)
    X = pd.concat((X, X_dummy), axis=1)
print(X.loc[5])
