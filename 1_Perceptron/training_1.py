#
#
#
import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()
sys.path.append(cwd)

from perceptron import Perceptron

# load data
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s,header=None,encoding='UTF-8')

# target
y = df.iloc[0:100,4].values # save species : 'Iris-setosa' / 'Iris-versicolor'
y = np.where(y=='Iris-setosa',0,1)	# if 'Iris-setosa' > 1, else > 0 : return numpy array
print(y)
# features
X = df.iloc[0:100,[0,2]].values # to numpy array # print(X.shape) : numpy.shape
# create perceptron
ppn = Perceptron(eta=0.1,n_iter=10)

ppn.fit(X,y)

# plot - 1 : check convergence
if 1 :
	plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Number of updates')
	plt.show()

# plot - 2

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

	markers = ('o','s','^','v','<')
	colors  = ('r','b','lightgreen','gray','cyan')
	cmap    = ListedColormap(colors[:len(np.unique(y))])	# probably only 2 colors

	# decision regions
	x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
	x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

	# this returns xvectors (xx1) / yvectors (xx2)
	xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
	                       np.arange(x2_min,x2_max,resolution))

	# print(np.arange(x1_min,x1_max,resolution).shape)
	# print(np.arange(x2_min,x2_max,resolution).shape)
	# print(xx1,xx1.shape)
	# print(xx2,xx2.shape)

	print(xx1.ravel().shape)
	print(xx2.ravel().shape)
	matrix = np.array([xx1.ravel(),xx2.ravel()])	#    2 x N array : number of feature '2'
	matrix = matrix.T                           	# to N x 2 array conversion
	print(matrix.shape)
	print(matrix) # this is 71675 x 2 : i.e., 71675 set
	#
	# Caution !
	# here, predict() of classifier uses the weight that obtained above !
	# lab (label)
	val = np.array([xx1.ravel(),xx2.ravel()]).T
	print('val:',val,val.shape)
	lab = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)	# 1d array # Transpose to (N, 2) form
	print('lab raw :',lab,lab.shape) # result of classifying 71675 data
	lab = lab.reshape(xx1.shape) # reshaping 1d like xx1.shape
	print('lab proc:',lab,lab.shape)

	plt.contourf(xx1,xx2,lab,alpha=0.3,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y == cl, 0], # in X, find row value where target y == cl for feature '0'
		            y = X[y == cl, 1],
		            alpha=0.4, c = colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor='black')

	#for idx, cl in enumerate(np.unique(y)):
	#	print(cl,np.where(y==cl,0,1))

	plt.show()

plot_decision_regions(X,y,ppn)















