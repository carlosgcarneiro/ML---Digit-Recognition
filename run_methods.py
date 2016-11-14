# Project to solve digit recognition:  https://www.kaggle.com/c/digit-recognizer/
#
# LogisticRegression References: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
#								 http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#								 http://scikit-learn.org/stable/modules/svm.html
#								 http://scikit-learn.org/stable/modules/ensemble.html#adaboost


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, svm
from sklearn.ensemble import AdaBoostClassifier
import csv


def read_file(file):
	with open(file, 'rb') as f:
		reader = csv.reader(f)
		data = list(reader)
		#data=[tuple(line) for line in csv.reader(f)]
	return np.array(data)

	
d = read_file('prepared_data.csv')
np.random.shuffle(d) # shuffle the data to train
X = d[:8000,1:]
Y = d[:8000,0]

#print ("Reading test...")
#X_test = read_file('test.csv')
#X_test = X_test[1:,:]
X_test = d[8001:9999,1:]
Y_test = d[8001:9999,0]

#logreg = linear_model.LogisticRegression(C=1e5)
clf_svm = svm.SVC()
clf_ada_boost = AdaBoostClassifier(n_estimators=100)


print ("Fitting...")
# we create an instance of Neighbours Classifier and fit the data.

#logreg.fit(X, Y)
clf_svm.fit(X, Y)
clf_ada_boost.fit(X, Y)


#for i in range(1,len(X_test)):
#	X_test[i] = map(int, X_test[i])	
	
Z_lr = logreg.predict(X_test)
Z_svm = clf_svm.predict(X_test)	
#Z_ada_boost = clf_ada_boost.predict(X_test)	

print Y_test
print Z_svm
print Z_ada_boost

