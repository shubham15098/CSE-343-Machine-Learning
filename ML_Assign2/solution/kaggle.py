import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import csv

with open('train.json') as json_data:
    d = json.load(json_data)

with open('test.json') as json_data:
    d2 = json.load(json_data)


x = []    
y = []
test_x = []
test_y = []


for i in d:
	y.append(i['Y'])
	i['X'] = str(" ".join(str(x) for x in i['X']))
	x.append(i['X'])
	
	


for i in d2:
	i['X'] = str(" ".join(str(x) for x in i['X']))
	test_x.append(i['X'])
	
	


#print(d[0].keys())
# print(y)
# print(x)

x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)


temp = TfidfVectorizer(ngram_range=(0,3), token_pattern=r"\b\w+\b",norm ='l2', binary=True)
x_n = temp.fit_transform(x)

clf = svm.LinearSVC(C=0.3529) #0.353
clf.fit(x_n, y)

x_f = []
y_f = []

y_new = []

#x_2 = temp.transform(x)
y_new = clf.predict(x_n)

for k in range(len(y)):
	if(y[k] == y_new[k]):
		x_f.append(x[k])
		y_f.append(y[k])


x_f2 = temp.fit_transform(x_f)
clf.fit(x_f2,y_f)


test_x = temp.transform(test_x)
test_y = clf.predict(test_x)



count = 1

with open('op.csv', 'wb') as file:
    
    file.write('Id,Expected\n')

    for j in test_y:
    	file.write(str(count)+','+str(j)+'\n')
    	count = count + 1

    


   