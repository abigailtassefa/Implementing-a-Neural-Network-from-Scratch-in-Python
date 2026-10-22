#implement a simple 3-layer neural network from scratch.
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#generate some random data for training
np.random.seed(0)
x,y=sklearn.datasets.make_moons(200,noise=0.20)
plt.scatter(x[:,0], x[:,1], s=40,c=y,cmap=plt.cm.Spectral)

            