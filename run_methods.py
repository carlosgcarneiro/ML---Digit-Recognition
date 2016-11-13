# Project to solve digit recognition:  https://www.kaggle.com/c/digit-recognizer/
#
# LogisticRegression References: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
#								 http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import csv

with open('prepared_data.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    #data=[tuple(line) for line in csv.reader(f)]
d = np.array(data)

np.random.shuffle(d) # shuffle the data to train

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

X = d[:,1:]
Y = d[:,0]
print ("Fitting...")
# we create an instance of Neighbours Classifier and fit the data.

logreg.fit(X, Y)

print ("Reading test...")
with open('test.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    #data=[tuple(line) for line in csv.reader(f)]
	
X_test = np.array(data)


for i in range(1,len(X_test)):
	X_test[i] = map(int, X_test[i])	
	
Z = logreg.predict(examples)

print Z

