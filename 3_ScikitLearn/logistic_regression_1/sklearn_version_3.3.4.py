import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

# data division : train / sets

from sklearn.model_selection import train_test_split

random_seed = 1

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = random_seed, stratify = y ) # second arg 'y' used for the stratification

print('y       label count : ', np.bincount(y))
print('y train label count : ', np.bincount(y_train))
print('y test  label count : ', np.bincount(y_test))

# data standardise

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train) # calculate feature avg. std.
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

# End of preparation

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
'''
	FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7.
	Use OneVsRestClassifier(LogisticRegression(..)) instead. Leave it to its default value to avoid this warning.

	FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7.
	From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
'''
lr = OneVsRestClassifier(LogisticRegression(C=100.0,solver='lbfgs'))			# C : over-fitting regularisation parameter
#lr = LogisticRegression(C=100.0, solver='lbfgs')

#lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')			# deprecated
#lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='multinomial')	# deprecated
#
# gradient descent method = lbfgs
# multi-class classification : ovr (One-Versus-Rest) / other possible : 'multinomial' # deprecated
#

lr.fit(X_train_std,y_train)

from plot import plot_decision_regions

# using all data to show model accuracy
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined_std = np.hstack((y_train,y_test))

plot_decision_regions( X = X_combined_std, y = y_combined_std, classifier = lr,
					test_idx = range(y.shape[0]-y_test.shape[0],y.shape[0]) )

plt.xlabel('Petal length std')
plt.ylabel('Petal width std')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# get probability 

proba = lr.predict_proba(X_test_std[:3,:])
print(proba)
print(proba.argmax(axis=0))	# 0 column-wise axis=N N:[0:max-dimension] , no axis -> flattend (default)
print(proba.argmax(axis=1))	# 1 row-wise    axis=N N:[0:max-dimension] , no axis -> flattend (default)

proba = lr.predict(X_test_std[:3,:])
print(proba)

proba = lr.predict(X_test_std[10,:].reshape(1,-1)) # add dimension, e.g., [1 2 3] (3,) > [1 2 3].reshape(1,-1) > [[1 2 3]] (1,3)
print(proba)
