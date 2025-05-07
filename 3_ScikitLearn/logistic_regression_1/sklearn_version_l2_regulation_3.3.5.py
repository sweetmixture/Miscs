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

#
# L2 parameter : C > inverse of lambda regulation hyper parameter
#
# defualt C = 1.0
weights, params = [], []

for c in np.arange(-5,5):

	lr = OneVsRestClassifier(LogisticRegression(C=10.**c, solver='lbfgs'))
	lr.fit(X_train,y_train)

	# data structure/contents investigation
	#print(lr.estimators_[0].coef_[0].shape) # (2,)
	#print(lr.estimators_[0].coef_,lr.estimators_[0].coef_.shape)	# since OvR, weight length always 2 for any cases (1x2)

	#
	# * for demonstration use class 1 > coef 2
	#
	weights.append(lr.estimators_[1].coef_[0]) # append (1,2) as (2,) # chosen OvR case : class 1 vs (class 0, class 2)
	params.append(10.**c)

weights = np.array(weights)
print(weights,weights.shape) # (10,2)

plt.plot(params, weights[:,0], label='Petal length')
plt.plot(params, weights[:,1], label='Petal width', linestyle='--')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

# end of regularisation test ----------------------------------------------------------------------------------------------
