# -*- coding: utf-8 -*-

"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch

import torch.nn.parallel

import torch.utils.data


"""## Importing the dataset"""

# We won't be using this dataset.
#since data set is binary reading method is different
#fistly the path of the file, sperartor ,since name of the column are not specified in binary file headers are used to make default columnn names
#encoding is used for the special characters in the movie tiltes.
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

"""## Preparing the training set and the test set"""

#converting into numpy array and it is an 80% 20% test split
#u1.base is training set containg 80% of the combined data of users rating and movie
#u1.test is the training set of the combines data
#file is in the form of index,user,movie,ratings,time stamps(not reqd)
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

"""## Getting the number of users and movies"""
#this is done so that 2 matrices can be made one for training set and other for the test set
#each row in the matrix will represent a  user and each column a movie
#the cells in matrix will denote the rating and if in case a user has not watched a partucular mavie the cell will contain 0
#both taring and test  set have same number of users ie. 943
#since we have to take all the users that is why maximum number of users id will denote the maxinumum no. of users 
#hence max() is used and it has to converted into integer that is why int().
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

"""## Converting the data into an array with users in lines and movies in columns"""
#creating a structure that will contain these observation that will go into the network
#and their different features that are going to be there in input nodes
#observations in rows and features in columns.
#here we are creating lists in lists...the first list corresponds to first user and so on
#each lists contains rating by the user for each movie.
def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    #takes all the movies of the first user.
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    #takes all the ratings of the first user.
    ratings = np.zeros(nb_movies)
    #initializing the ratings of all the movies in nb_movies data set to 0.
    ratings[id_movies - 1] = id_ratings
    #the movies whose rating has been found will be made equal to the ratong given by the user and others will remain to 0 so that we'll find out what all movies have not rated by the users
    new_data.append(list(ratings))
    #all the movies with or without ratings are apended to the new_data array.
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)
#list of 943 users have been created which contains individual lists at all the indices which further contains ratings of all the movies rated by that particular user

"""## Converting the data into Torch tensors"""

#for crating architechture of neural netwrok pytorch is used
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

"""## Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)"""

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
#done for both training and test sets
#if rating is 1 or 2 the movie is not liked
#if rating is 3 or more than 3 movie is liked
#if rating is 0 i.e not rated by the user, than we denote it using -1.

"""## Creating the architecture of the Neural Network"""

#an RBM is a probability graphical model and it is built using a class(as it contains various models)
#all classes must have init method to initilize the parameters.
class RBM():
   #init method is invoked automatically when the method is invoked
  def __init__(self, nv, nh):
      #self is default arg. and coressponds to the object that will be created afterwords
      #nv is the number of visible nodes and nh is the nuber of hidden nodes
    self.W = torch.randn(nh, nv)   #randn() is used to initialize the weights randomly  according to a normal distribution
    self.a = torch.randn(1, nh)    #bias for probability of hidden nodes
    self.b = torch.randn(1, nv)    #bias for probability of visible nodes
    
  def sample_h(self, x):           #x will coresspond to isible nodes
      #sampling of hiden nodes and visible nodes.
    wx = torch.mm(x, self.W.t())   #mm() is to make product of 2 tensors
    activation = wx + self.a.expand_as(wx)
    p_h_given_v = torch.sigmoid(activation)
    return p_h_given_v, torch.bernoulli(p_h_given_v)

  def sample_v(self, y):
      #provides probability of visible nodes given hidden nodes.
    wy = torch.mm(y, self.W)
    activation = wy + self.b.expand_as(wy)
    p_v_given_h = torch.sigmoid(activation)
    return p_v_given_h, torch.bernoulli(p_v_given_h)

  def train(self, v0, vk, ph0, phk):
    #v0= input vector containing the rantings by 1 particular user
    #vk= visible nodes obtained after k samplings 
    #ph0= vector of probabilities that at the first iteration the hidden nodes equla one giveb the values of v0
    #phk= probabilities of hidden nodes after k sampling given the values of visible node vk
    #updation of weights and biases according to the formula.
    self.W = self.W+(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    self.b = self.b+torch.sum((v0 - vk), 0)
    self.a = self.a+torch.sum((ph0 - phk), 0)
    
nv = len(training_set[0])   #training_set[0] coressponds to first line of the training setand length coressponds to all the elements in the first line
nh = 100                    #can choose any number.
batch_size = 100            #any number (larger batch size=better performance results)
rbm = RBM(nv, nh)           

"""## Training the RBM"""

nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
  train_loss = 0        
  s = 0.
  
  for id_user in range(0, nb_users - batch_size, batch_size):
    #vk is input batch of the ratings of the movies
    #vo is same just that it is a target
    vk = training_set[id_user : id_user + batch_size]
    v0 = training_set[id_user : id_user + batch_size]
    ph0,_ = rbm.sample_h(v0)
    #,_ gets the first element of the returned value
    
    for k in range(10):
        #sampling hidden nodes from all other nodes
      _,hk = rbm.sample_h(vk)
      _,vk = rbm.sample_v(hk)
      vk[v0<0] = v0[v0<0]     #retaining all the ratings which were -1.
      
    phk,_ = rbm.sample_h(vk)
    rbm.train(v0, vk, ph0, phk)
    train_loss =train_loss + torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
    s =s + 1.
  print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

"""## Testing the RBM"""

test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    #v is the input set that will be used to activate hidden neurons to get the output
    vt = test_set[id_user:id_user+1]
    #vt contains original ratings of the test set which can we used to make predictions in the test set
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss = test_loss + torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s =s + 1.
print('test loss: '+str(test_loss/s)) 

u = np.random.choice([0,1], 100000)
v = np.random.choice([0,1], 100000)
u[:50000] = v[:50000]
sum(u==v)/float(len(u)) # -> you get 0.75
np.mean(np.abs(u-v)) # -> you get 0.25