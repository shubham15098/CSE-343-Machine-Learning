import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing

with h5py.File("dataset_partA.h5", 'r') as hf:
  x = hf['X'][:]
  Y = hf['Y'][:]


x = np.array(x)
Y = np.array(Y)

for k in range(len(Y)):
  if(Y[k] == 7):
    Y[k] = 0
  else:
    Y[k] = 1


X = []
temp = []
temp= np.array(temp)

for k in range(len(Y)):
  temp = []
  for j in range(28):
    temp.extend(x[k][j])
  X.append(temp)


X = np.array(X)
y = np.array(Y)

# X = preprocessing.scale(X)
X = X.astype(float)
mean1 = np.mean(X, axis = 0)
std1 = np.std(X, axis = 0)
std1 = std1.astype(float)
mean1 = mean1.astype(float)
# print std1
X = (X - mean1) 
X = X.astype(float)

for s in range(len(std1)):
  for z in range(len(X)):
    if(std1[s] != 0):
      X[z][s] = X[z][s] / std1[s] 


#--------------------------------------------------------------------------------

D = len(X[0])
K = 2

n1 = int(input("Enter num of layers, e.g. 784,100,50,10; here number of layers will be 2- "))

# h contains size of hidden layer
h = []
for k1 in range(n1):  
  temp = int(input())
  h.append(temp)

print h


w_array = []
b_array = []

# initialize parameters randomly
w_array.append(0.01 * np.random.randn(D,h[0]))
b_array.append(np.zeros((1,h[0])))
print w_array
# continue

for k1 in range(len(h)-1):
  w_array.append(0.01 * np.random.randn(h[k1],h[k1+1]))
  b_array.append(np.zeros((1,h[k1+1])))

w_array.append(0.01 * np.random.randn(h[len(h)-1],K)) 
b_array.append(np.zeros((1,K)))


# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

hidden_layer = []
dw_array = []
db_array = []
dhidden_array = []
# gradient descent loop
num_examples = X.shape[0]
for i in xrange(10000):
  hidden_layer = []
  dw_array = []
  db_array = []
  dhidden_array = []
  # evaluate class scores, [N x K]
  hidden_layer.append(np.maximum(0, np.dot(X, w_array[0]) + b_array[0])) # note, ReLU activation
  for k2 in range(len(h)-1):
    hidden_layer.append(np.maximum(0, np.dot(hidden_layer[k2], w_array[k2+1]) + b_array[k2+1]))


  scores = np.dot(hidden_layer[len(hidden_layer)-1], w_array[len(w_array)-1]) + b_array[len(b_array)-1]
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  p2 = 1 
  data_loss = np.sum(corect_logprobs)/num_examples
  p2 = 2
  reg_loss = 0
  p2 = 3

  for k3 in range(len(w_array)):
    reg_loss = reg_loss + (0.5*reg*np.sum(w_array[k3]*w_array[k3]))


  loss = data_loss + reg_loss
  
  if i % 200 == 0:
    p2 = 3
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  p2 = 4
  dscores[range(num_examples),y] -= 1
  p2 = 5
  dscores /= num_examples
  p2 = 6
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dw_array.append(np.dot(hidden_layer[len(hidden_layer)-1].T, dscores))
  db_array.append(np.sum(dscores, axis=0, keepdims=True))
  
  

  # next backprop into hidden layer
  dhidden_array.append(np.dot(dscores, w_array[len(w_array)-1].T))

  # backprop the ReLU non-linearity
  dhidden_array[0][hidden_layer[len(hidden_layer)-1] <= 0] = 0
  
  # finally into W,b
  dw_array.append(np.dot(hidden_layer[len(hidden_layer)-2].T, dhidden_array[0]))
  db_array.append(np.sum(dhidden_array[0], axis=0, keepdims=True))
  

  for k4 in range(len(hidden_layer)-2):
    dhidden_array.append(np.dot(dhidden_array[k4], w_array[len(w_array)-1-(k4+1)].T))
    # change the one i have appended jus above
    dhidden_array[len(dhidden_array)-1][hidden_layer[len(hidden_layer)-1-(k4+1)] <= 0] = 0

    dw_array.append(np.dot(hidden_layer[len(hidden_layer)-(3+k4)].T, dhidden_array[len(dhidden_array)-1]))
    db_array.append(np.sum(dhidden_array[len(dhidden_array)-1], axis=0, keepdims=True))

  # next backprop into hidden layer
  
    # finally into W,b
  dhidden_array.append(np.dot(dhidden_array[len(dhidden_array)-1], w_array[1].T))
  dhidden_array[len(dhidden_array)-1][hidden_layer[0] <= 0] = 0
  dw_array.append(np.dot(X.T, dhidden_array[len(dhidden_array)-1]))
  db_array.append(np.sum(dhidden_array[len(dhidden_array)-1], axis=0, keepdims=True))

  jk = 0
  for jk in range(len(dw_array)):
    # print jk
    dw_array[jk] += reg * w_array[len(w_array)-(1+jk)]

  for jk in range(len(w_array)):
    w_array[jk] += -step_size * dw_array[len(dw_array)-(1+jk)]
    b_array[jk] += -step_size * db_array[len(db_array)-(1+jk)]




    
  


# we will go forward one last time to predict output..this time we use our best thetas

# hidden_layer1 = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
# hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W2) + b2) # note, ReLU activation
# scores = np.dot(hidden_layer2, W3) + b3

# predicted_class = np.argmax(scores, axis=1)
# print 'training accuracy: %.2f' % (np.mean(predicted_class == y))