# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#mport sklearn
#rom sklearn import datasets

#Objectives: Using Panda to get the data 
# Make a chart of the data
# Use unsupervised learning voodoo magic to make predictions

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

spotify_train = "../input/trainandtest2electicboogaloo/spotify_train.csv"
spotify_x_test = "../input/trainandtest2electicboogaloo/spotify_x_test.csv"


spotify_train = pd.read_csv(spotify_train)
spotify_test = pd.read_csv(spotify_x_test)

x_train = spotify_train[['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']].astype(float)
y_train = spotify_train[['genre']].astype(int)

x_test = spotify_test[['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']].astype(float)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier(n_estimators=128)
rfc_model.fit(x_train, np.ravel(y_train))
rfc_y_pred = rfc_model.predict(x_test)


rfc_pred_csv = pd.DataFrame({'id': range(len(rfc_y_pred)), 'genre': rfc_y_pred})
rfc_pred_csv.to_csv('submission.csv', index=False)


print("Fin.")
