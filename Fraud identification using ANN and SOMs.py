# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:34:37 2020

@author: pc
"""

#hybrid deep learning model using ANN and SOMs

#PART 1 SOMs

import numpy as np
import pandas as pd


dataset=pd.read_csv("Credit_Card_Applications.csv")
X=dataset.iloc[:,:-1].values  #all columns except last
y=dataset.iloc[:,-1].values   #only last column

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0, 1))
#feature scaling using normalisation technique
X=sc.fit_transform(X)

#Training SOM
from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
#input length is athe number of attributes in X
#sigma is the radius of different neighbourhood in the grid
#learning rate decides by how much weights are updated during each iteration so higher the learnig rate fater will be the convergernce land lower the leaning rate the longer time will be taken
#decay function is to improve the convergence
#10X10 grid ; default value for input_len and learning_rate
 
som.random_weights_init(X);
#initializing weights

som.train_random(data=X,num_iteration=100)
#method to train som on X

#plotting SOMs
#MID is the mean interneuron diatance of winning nodes from neurons in its neighbourhood
#higher the MID, the neurons will be far away from winning node in its neighbourhood
#higher MID nodes will be the outliers because they will be away and different from other nodes
from pylab import bone,colorbar,plot,show,pcolor
bone() 
#used for initilaixzing the window of output

pcolor(som.distance_map().T)
#Adding colours to all the winning nodes of SOMs

#Now we have to know which colour coressponds to higher or lower MID
colorbar()
#highest MID is white colour and lowest is White colour

#now we add some markers to correctly distinguish between fraud and other customers
#green-approved
#red-not approved
markers = ['o', 's']  #S for approved and o for not approved
colors = ['r', 'g']   #coressponding colour for markers

#associating markers and colours
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings=som.win_map(X)
#Finding the frauds in the dataset

frauds = np.concatenate((mappings[(7,5)],mappings[(8,5)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
  
  
#PART 2 ANN

#create metrics of features
customers=dataset.iloc[:,1:].values  #all columns except first
#independent variable and it includes last column as it helps in determining the fraud factor

#creation of dependent variable
is_farud=np.zeros(len(dataset)) 

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_farud[i]=1                 #the dependent variable vector will be made equal to 1 for each of the fraud customer id.

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=2,kernel_initializer='uniform',activation='relu',input_dim=15))

classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(customers,is_farud,batch_size=1,epochs=4)

y_pred=classifier.predict(customers)

y_pred=frauds = np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis = 1)
#for vertival concatenation axis =0 and for horizontal concatenation axis=1

y_pred=y_pred[y_pred[:,1].argsort()]

