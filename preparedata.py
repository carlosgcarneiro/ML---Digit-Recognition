# Project to solve digit recognition:  https://www.kaggle.com/c/digit-recognizer/
#
# LogisticRegression References: http://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
#								 http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


import numpy as np
import csv
import sys

qtd = int(sys.argv[1])

print ("Preparing data...")
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    data = list(reader)
    #data=[tuple(line) for line in csv.reader(f)]
d = np.array(data)

d = d[1:,:] # d quiting the header

np.random.shuffle(d)

# dY where d is Y
d0 = d[np.where(d[:,0] == '0')] 
d0 = d0[:qtd,:]

d1 = d[np.where(d[:,0] == '1')]
d1 = d1[:qtd,:]

d2 = d[np.where(d[:,0] == '2')]
d2 = d2[:qtd,:]

d3 = d[np.where(d[:,0] == '3')]
d3 = d3[:qtd,:]

d4 = d[np.where(d[:,0] == '4')]
d4 = d4[:qtd,:]

d5 = d[np.where(d[:,0] == '5')]
d5 = d5[:qtd,:]

d6 = d[np.where(d[:,0] == '6')]
d6 = d6[:qtd,:]

d7 = d[np.where(d[:,0] == '7')]
d7 = d7[:qtd,:]

d8 = d[np.where(d[:,0] == '8')]
d8 = d8[:qtd,:]

d9 = d[np.where(d[:,0] == '9')]
d9 = d9[:qtd,:]




with open('prepared_data.csv', 'wb') as file:
	csv_writer = csv.writer(file) 
	for i in range(0,qtd):
		csv_writer.writerow(map(int, d0[i,:]))
		csv_writer.writerow(map(int, d1[i,:]))
		csv_writer.writerow(map(int, d2[i,:]))
		csv_writer.writerow(map(int, d3[i,:]))
		csv_writer.writerow(map(int, d4[i,:]))
		csv_writer.writerow(map(int, d5[i,:]))
		csv_writer.writerow(map(int, d6[i,:]))
		csv_writer.writerow(map(int, d7[i,:]))
		csv_writer.writerow(map(int, d8[i,:]))
		csv_writer.writerow(map(int, d9[i,:]))

		
print ("DONE! Data prepared.")