# one vs one
import os.path
import argparse
import h5py
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def pred_method(mesh,X,Y,check_assign):
	
		# find the labels
	label= []
	d = {}

	if(check_assign == 2):
		for i in y:
			if i in d:
				continue
			else:
				d[i] = 69

	else:
		for i in Y_array1:
			if i in d:
				continue
			else:
				d[i] = 69
		for i in Y_array2:
			if i in d:
				continue
			else:
				d[i] = 69



	label = d.keys()
	label.sort()
	final = np.zeros((len(mesh),len(label)))

	for k1 in range(len(label)-1):
			for k2 in range(k1+1,len(label)):
				new_trainX = []
				new_trainY = []
				first = label[k1]
				second = label[k2]
				
				for i1 in range(len(X)):
					if(Y[i1] == first or Y[i1] == second):
						new_trainX.append(X[i1])
						if(Y[i1] == first):

							new_trainY.append(1)
						elif(Y[i1] == second):
							new_trainY.append(-1)


				
				
				clf = svm.SVC(kernel='linear')
				clf.fit(new_trainX, new_trainY)  

				z = clf.dual_coef_
				s = clf.support_vectors_
				intercept = clf.intercept_

				z = np.array(z)
				s = np.array(s)
				#intercept = np.array(intercept)
				test_x = np.array(X)

				for j1 in range(len(mesh)):
					sum11 = 0
					for j2 in range(len(s)):
						sum11 = sum11 + (np.dot(s[j2],mesh[j1]))*z[0][j2]

					sum11 = sum11 + intercept[0]
					
					if(sum11 > 0):
						final[j1][first] = final[j1][first] + 1
					elif(sum11 < 0):
						final[j1][second] = final[j1][second] + 1


		


		#prediciton
	my_pred = []



	for i1 in range(len(final)):
		max11 = -1000000
		index11 = -100000
		for i2 in range(len(final[0])):
			if(final[i1][i2] >= max11):
				max11 = final[i1][i2]
				index11 = i2

		my_pred.append(index11)

	my_pred = np.array(my_pred)
	return my_pred
















# Load the data
with h5py.File("Data/data_5.h5", 'r') as hf: #part_A_train
	x = hf['x'][:] # make it 'X' and 'Y' in case of assign 1
	y = hf['y'][:]

check_assign = int(input("Choose assign num: ")) # make it 1 for assign 1 else 2


# k-folds ; for k = 5 
arr1 = []
arr2 = []

X_array1 = []
X_array2 = []
X_array3 = []
X_array4 = []
X_array5 = []

Y_array1 = []
Y_array2 = []
Y_array3 = []
Y_array4 = []
Y_array5 = []

k = 5


end_point1 = len(x)/5
end_point2 = end_point1 + end_point1
end_point3 = end_point2 + end_point1
end_point4 = end_point3 + end_point1
end_point5 = end_point4 + end_point1

for i in range(0,end_point1):
	X_array1.append(x[i])

	if(check_assign == 2):
		Y_array1.append(y[i])

	else:
		for j in range(len(y[0])):
			if(y[i][j] == 1):
				Y_array1.append(j)
				break

	


for i in range(end_point1,end_point2):
	X_array2.append(x[i])
	
	if(check_assign == 2):
		Y_array2.append(y[i])
	else:
		for j in range(len(y[0])):
			if(y[i][j] == 1):
				Y_array2.append(j)
				break

	

for i in range(end_point2,end_point3):
	X_array3.append(x[i])

	if(check_assign == 2):
		Y_array3.append(y[i])

	else:
		for j in range(len(y[0])):
		
			if(y[i][j] == 1):
				Y_array3.append(j)
				break

	

for i in range(end_point3,end_point4):
	X_array4.append(x[i])

	if(check_assign == 2):
		Y_array4.append(y[i])
	else:
		for j in range(len(y[0])):
			if(y[i][j] == 1):
				Y_array4.append(j)
				break
		


for i in range(end_point4,end_point5):
	X_array5.append(x[i])

	if(check_assign == 2):
		Y_array5.append(y[i])

	else:
		for j in range(len(y[0])):
			if(y[i][j] == 1):
				Y_array5.append(j)
				break


# find the labels
label= []
d = {}

if(check_assign == 2):
	for i in y:
		if i in d:
			continue
		else:
			d[i] = 69

else:
	for i in Y_array1:
		if i in d:
			continue
		else:
			d[i] = 69
	for i in Y_array2:
		if i in d:
			continue
		else:
			d[i] = 69



label = d.keys()
label.sort()

# confusion matrix
cm = np.zeros((len(label),len(label)))

#print(Y_array4[9])
#print(Y[2529])

# we will take one out of five above array to test data, rest will be used for training
# array1 to test

for itr in range(5):
	if(itr == 0):
		s1X = X_array2 + X_array3 + X_array4 + X_array5
		s1Y = Y_array2 + Y_array3 + Y_array4 + Y_array5
		test_x = X_array1 # i have used test_x and s1 etc...therefore change their values for k-fold
		test_y = Y_array1
	
	elif(itr == 1):
		s1X = X_array1 + X_array3 + X_array4 + X_array5
		s1Y = Y_array1 + Y_array3 + Y_array4 + Y_array5
		test_x = X_array2 # i have used test_x and s1 etc...therefore change their values for k-fold
		test_y = Y_array2

	elif(itr == 2):
		s1X = X_array2 + X_array1 + X_array4 + X_array5
		s1Y = Y_array2 + Y_array1 + Y_array4 + Y_array5
		test_x = X_array3 # i have used test_x and s1 etc...therefore change their values for k-fold
		test_y = Y_array3

	elif(itr == 3):
		s1X = X_array2 + X_array3 + X_array1 + X_array5
		s1Y = Y_array2 + Y_array3 + Y_array1 + Y_array5
		test_x = X_array4 # i have used test_x and s1 etc...therefore change their values for k-fold
		test_y = Y_array4

	elif(itr == 4):
		s1X = X_array2 + X_array3 + X_array4 + X_array1
		s1Y = Y_array2 + Y_array3 + Y_array4 + Y_array1
		test_x = X_array5 # i have used test_x and s1 etc...therefore change their values for k-fold
		test_y = Y_array5

	
	C = [1,0.1,10]

	#make the final matrix
	final = np.zeros((len(test_x),len(label)))

	for c in C:

		# here we are taking combinations of label..like 0,1; 0,2; 1,2
		for k1 in range(len(label)-1):
			for k2 in range(k1+1,len(label)):
				new_trainX = []
				new_trainY = []
				first = label[k1]
				second = label[k2]
				
				for i1 in range(len(s1X)):
					if(s1Y[i1] == first or s1Y[i1] == second):
						new_trainX.append(s1X[i1])
						if(s1Y[i1] == first):

							new_trainY.append(1)
						elif(s1Y[i1] == second):
							new_trainY.append(-1)


				
				
				clf = svm.SVC(kernel='linear', C=c)
				clf.fit(new_trainX, new_trainY)  

				z = clf.dual_coef_
				s = clf.support_vectors_
				intercept = clf.intercept_

				z = np.array(z)
				s = np.array(s)
				#intercept = np.array(intercept)
				test_x = np.array(test_x)

				for j1 in range(len(test_x)):
					sum11 = 0
					for j2 in range(len(s)):
						sum11 = sum11 + (np.dot(s[j2],test_x[j1]))*z[0][j2]

					sum11 = sum11 + intercept[0]
					
					if(sum11 > 0):
						final[j1][first] = final[j1][first] + 1
					elif(sum11 < 0):
						final[j1][second] = final[j1][second] + 1


		


		#prediciton
		my_pred = []



		for i1 in range(len(final)):
			max11 = -1000000
			index11 = -100000
			for i2 in range(len(final[0])):
				if(final[i1][i2] >= max11):
					max11 = final[i1][i2]
					index11 = i2

			my_pred.append(index11)		

		# print(my_pred)
		# print(test_y)

		c11 = 0
		for i1 in range(len(test_y)):
			if(c == 1):
				cm[my_pred[i1]][test_y[i1]] = cm[my_pred[i1]][test_y[i1]] + 1
			if(my_pred[i1] == test_y[i1]):
				c11 = c11 + 1

		acc = (c11/float(len(test_y)))*100

		print('k = '+(str(itr+1))+', C = ' +str(c))
		print(acc)

print(cm)









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

# width = 0.02

# x_min,x_max = x[:,0].min()-1, x[:,0].max()+1
# y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
# xx,yy = np.meshgrid(np.arange(x_min,x_max,width),np.arange(y_min,y_max,width))
# z = pred_method(np.c_[xx.ravel(),yy.ravel()],x,y,check_assign)
# z = np.array(z)
# z=z.reshape(xx.shape)
# plt.pcolormesh(xx,yy,z,cmap=plt.cm.Paired)
# plt.scatter(x[:,0],x[:,1],c=y)
# # plt.savefig('plots_with_decsionboundary'+user_input[5]+'png')
# plt.show()