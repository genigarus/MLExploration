# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:10:16 2018

@author: gmahato
"""

# From Unsupervised learning SOM: get all potential fraudulent clients
## SOM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#feature scaling
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#train model
from minisom import MiniSom
fraud_som = MiniSom(x = 10,y = 10, input_len = 15)
fraud_som.random_weights_init(X)
fraud_som.train_random(X, 100)

#visualization
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(fraud_som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = fraud_som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# find frauds
fraud_mappings = fraud_som.win_map(X)
frauds = np.concatenate((fraud_mappings[3,7], fraud_mappings[5,7]))
frauds = sc.inverse_transform(frauds)

# save to file the details of potential fraudulent customers
df = pd.DataFrame(frauds)
df.to_csv('fraud_detection_results.csv')


## From Supervised Learning: ANN get probability of fraud

# create features
customers = dataset.iloc[:, 1:].values

# create dependent variable
fraud_results = np.zeros((dataset.shape[0]))
for i in range(dataset.shape[0]):
    if dataset.iloc[i,0] in frauds:
            fraud_results[i] = 1


#feature scaling
from sklearn.preprocessing import StandardScaler
sup_sc = StandardScaler()
customers = sup_sc.fit_transform(customers, fraud_results)
fraud_results = sup_sc.transform(fraud_results)

# model creation
from keras.models import Sequential
from keras.layers import Dense

input_dim = customers.shape[1]
classifier_model = Sequential()
classifier_model.add(Dense(3, activation='relu', input_dim=input_dim))
classifier_model.add(Dense(1, activation='relu'))

classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier_model.fit(customers, fraud_results, batch_size=1, epochs=5)

# get probability of frauds by each customer
fraud_probab = classifier_model.predict(customers)
fraud_probab = np.concatenate((dataset.iloc[:, 0:1], fraud_probab), axis=1)
fraud_probab = fraud_probab[fraud_probab[:, 1].argsort()]
