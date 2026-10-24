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
nn_input_dim = 2 # input layer dimensionality based on the dataset's dimensionality
nn_output_dim = 2 # output layer dimensionality based on the number of classes 

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

#function to calculate the loss for the neural network
def calculate_loss(model):
    w1,b1,w2,b2= model['W1'], model['b1'], model['W2'], model['b2']

    #Forward peopagation 
    z1=x.dot(w1)+b1
    a1=np.tanh(z1)
    z2=a1.dot(w2)+b2 #raw output
    exp_scores=np.exp(z2) #softmax activation step 1
    probs=exp_scores/np.sum(exp_scores, axis=1, keepdims=True) #softmax step 2

    #calculating the loss using negative log likelihood or cross entropy
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

     # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    return 1./num_examples * data_loss

def predict(model,x):
     w1,b1,w2,b2= model['W1'], model['b1'], model['W2'], model['b2']
     #Forward peopagation 
     z1=x.dot(w1)+b1
     a1=np.tanh(z1)
     z2=a1.dot(w2)+b2 #raw output
     exp_scores=np.exp(z2) #softmax activation step 1
     probs=exp_scores/np.sum(exp_scores, axis=1, keepdims=True) #softmax step 2
     return np.argmax(probs, axis=1)

#the training function for the neural network
#nn_hdim: Number of nodes in the hidden layer
# num_passes: Number of passes through the training data for gradient descent
# print_loss: If True, print the loss every 1000 iterations

def build_model(nn_hdim,num_passes=20000, print_loss=False):
     np.random.seed(0)
     w1= np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
     b1= np.zeros((1, nn_hdim))
     w2= np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
     b2= np.zeros((1, nn_output_dim))
     model = {}

    #Gradient decent for each batch 
     for i in xrange(0,num_passes):
          
          #forward prop
          z1=X.dot(w1)+b1
          a1=np.tanh(z1)
          z2=a1.dot(w2)+b2
          exp_scores= np.exp(z2)
          probs=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)

          #backprop
          #see math
          delta3=probs #output layer error
          delta3[range(num_examples),y]-=1
          dw2=(a1.T).dot(delta3)
          db2=np.sum(delta3, axis=0, keepdims=True)
          delta2=delta3.dot(w2.T) * (1 - np.power(a1, 2))
          dw1=np.dot(X.T, delta2)
          db1=np.sum(delta2, axis=0)
          

          


     

