import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets

# code is reffered from official scikit documentation
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html


def rbfKernel(X1,X2):
	# here we are making kernel matrix[t1,t2]
	t1 = X1.shape[0]
	t2 = X2.shape[0]

	#now we will initialize our kernel array and put zeros in it for now
	kernel_matrix = np.zeros((t1,t2))

	i = -1
	j = -1

	for x1 in X1:
		i = i + 1
		for x2 in X2:
			j = j + 1
			kernel_matrix[i,j] = np.exp(-0.7*np.linalg.norm(x1-x2)**2)

		j = -1


	return kernel_matrix



# Load the test data
with h5py.File("Data/data_5.h5", 'r') as hf:
	x = hf['x'][:]
	y = hf['y'][:]


clf = svm.SVC(kernel=rbfKernel,C=1.0)
clf.fit(x, y) 

x = np.array(x)
y = np.array(y)

first_column = []
second_column = []

for i in range(len(x)):
    first_column.append(x[i][0])
    second_column.append(x[i][1])

first_column = np.array(first_column)
second_column = np.array(second_column)

fc_min = 100000000
fc_max = -1000000
sc_min = 1000000
sc_max = -100000

for i in range(len(x)):
    if(first_column[i] > fc_max):
        fc_max = first_column[i]
    if(first_column[i] < fc_min):
        fc_min = first_column[i]

    if(second_column[i] > sc_max):
        sc_max = second_column[i]
    if(second_column[i] < sc_min):
        sc_min = second_column[i]

fc_min = fc_min - 1
sc_min = sc_min - 1
sc_max = sc_max + 1
fc_max = fc_max + 1


xx = np.arange(fc_min,fc_max,0.02)
yy = np.arange(sc_min,sc_max,0.02)

xx , yy = np.meshgrid(xx,yy)

xx = np.array(xx)
yy = np.array(yy)

xx1 = xx.ravel()
yy1 = yy.ravel()

ax=plt
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 
ax.scatter(first_column, second_column, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.xlim(xx.min(), xx.max())
ax.ylim(yy.min(), yy.max())
ax.title('SVM with kernel')

plt.show()

# x_min,x_max = x[:,0].min()-1, x[:,0].max()+1
# y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
# xx,yy = np.meshgrid(np.arange(x_min,x_max,width),np.arange(y_min,y_max,width))
# z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
# z = np.array(z)
# z=z.reshape(xx.shape)
# plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
# plt.scatter(x[:,0],x[:,1],c=y)
# # plt.savefig('plots_with_decsionboundary'+user_input[5]+'png')
# plt.show()



