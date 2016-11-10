# Project to solve digit recognition:  https://www.kaggle.com/c/digit-recognizer/
#
# LogisticRegression References: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
#								 http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


import numpy as np
import csv

with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    #data=[tuple(line) for line in csv.reader(f)]
d = np.array(data)

d = d[1:,:] # d quiting the header

np.random.shuffle(d)

# dY where d is Y
d0 = d[np.where(d[:,0] == '0')] 
d0 = d0[:1000,:]

d1 = d[np.where(d[:,0] == '1')]
d1 = d1[:1000,:]

d2 = d[np.where(d[:,0] == '2')]
d2 = d2[:1000,:]

d3 = d[np.where(d[:,0] == '3')]
d3 = d3[:1000,:]

d4 = d[np.where(d[:,0] == '4')]
d4 = d4[:1000,:]

d5 = d[np.where(d[:,0] == '5')]
d5 = d5[:1000,:]

d6 = d[np.where(d[:,0] == '6')]
d6 = d6[:1000,:]

d7 = d[np.where(d[:,0] == '7')]
d7 = d7[:1000,:]

d8 = d[np.where(d[:,0] == '8')]
d8 = d8[:1000,:]

d9 = d[np.where(d[:,0] == '9')]
d9 = d9[:1000,:]

print d0.shape


'''
Y = d[1:,0]
X = d[1:,1:] 

index,y = enumerate(Y)

count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0
count6 = 0
count7 = 0
count8 = 0
count9 = 0
with open('test1.csv', 'wb') as testfile:
    csv_writer = csv.writer(testfile)    
	for i in index:
		if (y[i]==0):
			csv_writer.writerow([0])
			count0 = 

#csv_writer.writerow([x[y] for x in hello])


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

print Y[0],Z[0]




# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
'''