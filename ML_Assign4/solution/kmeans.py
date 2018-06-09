# cluster size
import csv
import numpy as np
from sklearn import metrics

x = []
y = []
temp = []

with open('iris.csv','rb') as f:
	reader = csv.reader(f)

	for row in reader:
		temp = []
		l = len(row)
		t = row[l-1]
		y.append(t)

		for i in range(l-1):
			temp.append(row[i])

		x.append(temp)

		
		


# print len(x)
# print len(y)
x = np.array(x).astype(np.float)
y = np.array(y).astype(np.float)

k = int(input("enter num of clusters: "))

max1 = np.amax(x, axis=0)
min1 = np.amin(x, axis=0)


means = []
clusters = []
belongs2 = []
cluster_size = []

# initialize means
for i in range(k):
	z = np.zeros(len(x[0]))
	means.append(z)
	clusters.append([])
	cluster_size.append(0)

for i in range(len(x)):
	belongs2.append(-1)

# randomily assign means
for i in range(k):
	for j in range(len(x[0])):
		means[i][j] = np.random.uniform(min1[j],max1[j]);


count = -1
count2 = -1 

flag = 0

while(flag != 1):

	flag = 1
	count2 = -1
	count = -1
	
	for row in x:
		count2 = count2 + 1
		mini = 999999
		count = count + 1

		ind = -1

		for z in range(k):
			row = np.array(row)
			means[z] = np.array(means[z])
			# now i will find the distance between means[z] and row

			# temp_dis = np.linalg.norm(row, means[z])
			temp_dis = np.sqrt(np.sum((row-means[z])**2))

			if(temp_dis <= mini):
				mini = temp_dis
				ind = z
				



		# clusters[z].append(row)
		cluster_size[ind] = cluster_size[ind] + 1
		
		# update mean

		n = cluster_size[ind]
		
		new = ind

		if(ind != belongs2[count2]):
			flag = 0

		for kk in range(len(means[0])):
			m = means[ind][kk]
			m = float((m*(n-1)+row[i]))/n
			means[ind][kk] = round(m,3)


		belongs2[count2] = ind
		print ind


# final_clusters = []

# for i in range(k):
# 	final_clusters.append([])

# for row in x:



for item in x:
	mini = -999999
	ind = -1

	for z in range(k):
		temp_dis = np.sqrt(np.sum((row-means[z])**2))
		if(temp_dis <= mini):
			mini = temp_dis
			ind = z


	clusters[z].append(row)
        #Classify item into a cluster
        # index = Classify(means,item);

        #Add item to cluster
        # clusters[index].append(item);

    



for i in clusters:
	print i
	print "\n"

# print metrics.adjusted_rand_score(y, belongs2)
# print metrics.normalized_mutual_info_score(y, belongs2)
# print metrics.adjusted_mutual_info_score(y, belongs2)






