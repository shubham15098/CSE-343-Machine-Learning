# in classififer change the value of c 
import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt



# Load the data
with h5py.File("Data/data_4.h5", 'r') as hf: #part_A_train
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
	C = [1,0.1,10]


	#make the final matrix
	final = np.zeros((len(test_x),len(label)))

	#map to 1 and -1 and make new arrays

	for c in C:
		for k in label:
			new_trainX = []
			new_trainY = []
			
			for p in range(len(s1X)):

				if(s1Y[p] == k):
					new_trainY.append(1)
					new_trainX.append(s1X[p])
				else:
					new_trainY.append(-1)
					new_trainX.append(s1X[p])

			
			clf = svm.SVC(kernel='linear', C = c)
			clf.fit(new_trainX, new_trainY)  

			z = clf.dual_coef_
			s = clf.support_vectors_
			intercept = clf.intercept_

			z = np.array(z)
			s = np.array(s)
			#intercept = np.array(intercept)
			test_x = np.array(test_x)
			
			
			for i1 in range(len(test_x)):
				sum11 = 0
				for i2 in range(len(s)):
					sum11 = sum11 + (np.dot(s[i2],test_x[i1]))*z[0][i2]

				sum11 = sum11 + intercept[0]
				final[i1][k] = sum11

		#print (len(final[0]))

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
			if(my_pred[i1] == test_y[i1]):
				c11 = c11 + 1

		acc = (c11/float(len(test_y)))*100

		print('k = '+(str(itr+1))+', C = ' +str(c))
		print(acc)











