import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Load the test data
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
print Y

X = []
temp = []
temp= np.array(temp)

for k in range(len(Y)):
  temp = []
  for j in range(28):
    temp.extend(x[k][j])
  X.append(temp)


X = np.array(X)
Y = np.array(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,50), activation='logistic',max_iter=10000,
  verbose=10, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(X_train,y_train)



predictions = mlp.predict(X_test)
ans= accuracy_score(y_test,predictions)
print ans