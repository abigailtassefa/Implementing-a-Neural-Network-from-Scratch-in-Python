#implement a simple 3-layer neural network from scratch.
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#generate some random data for training
np.random.seed(0)
x,y=sklearn.datasets.make_moons(200,noise=0.20)
plt.scatter(x[:,0], x[:,1], s=40,c=y,cmap=plt.cm.Spectral)


#Implementation of the neural network

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
