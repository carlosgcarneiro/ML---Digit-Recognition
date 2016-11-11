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

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 1:].min() - .5, X[:, 1:].max() + .5
y_min, y_max = X[:, 0].min() - .5, X[:, 0].max() + .5
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