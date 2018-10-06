# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
frauds = np.concatenate((fraud_mappings[6,1], fraud_mappings[7,1], fraud_mappings[2,7], fraud_mappings[2,8]))
frauds = sc.inverse_transform(frauds)
df = pd.DataFrame(frauds)
df.to_csv('fraud_detection_results.csv')