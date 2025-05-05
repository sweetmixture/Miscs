#
#
#
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
	# Caution !
	# here, predict() of classifier uses the weight that obtained above !
	# lab (label)
	val = np.array([xx1.ravel(),xx2.ravel()]).T
	lab = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)	# 1d array # Transpose to (N, 2) form
	lab = lab.reshape(xx1.shape) # reshaping 1d like xx1.shape

	plt.contourf(xx1,xx2,lab,alpha=0.3,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x = X[y == cl, 0], # in X, find row value where target y == cl for feature '0'
		            y = X[y == cl, 1],
		            alpha=0.4, c = colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor='black')
