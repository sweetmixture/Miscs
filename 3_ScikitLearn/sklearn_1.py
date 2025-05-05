from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

print('class label:',np.unique(y)) # [0,1,2] > Iris-setosa, Iris-versicolor, Iris-virginica

# data division : train / test

from sklearn.model_selection import train_test_split # works for array-like object

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) # stratification

# result of startification
print('y       label count: ', np.bincount(y))
print('y train label count: ', np.bincount(y_train))
print('y test  label count: ', np.bincount(y_test))

# --------------------------------------------------

# feature scaling (standarisation)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#print(X_train)
sc.fit(X_train) # calculate feature avg. std.
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test) # here use same avg. std. obtained from X_train !

# Perceptron
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)

print(f'wrong classification : {(y_test != y_pred).sum()} out of {y_test.shape[0]}')

# or using sklearn metrics

from sklearn.metrics import accuracy_score

print(f'accuracy metrics : {accuracy_score(y_test, y_pred):>.3f}')		# use sklearn.metrics.acuracy_score method
print(f'accuracy ppn     : {ppn.score(X_test_std, y_test):>.3f}')		# use ppn (sklearn.linear_model.Perceptron) score method

