import sys
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from LogisticRegression import LogisticRegressionGD	# only works for binary classification problem
from plot import plot_decision_regions

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

print(y.shape)
print('class label:',np.unique(y)) # [0,1,2] > Iris-setosa, Iris-versicolor, Iris-virginica

# data division : train / test

from sklearn.model_selection import train_test_split # works for array-like object
rs = 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=int(f'{rs}'), stratify=y) # stratification
#                                                         ^^^^^^^^^ default 0.25
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
#print(y_train,y_train.shape)
#print(y_train == 0 )
#print(y_train == 1 )
#print((y_train == 0) | (y_train == 1))
X_train_01_subset = X_train_std[ (y_train == 0) | (y_train == 1) ] # y_train == 0 returns list of Trues/Falses : collect only target 0 or 1
y_train_01_subset = y_train[ (y_train == 0) | (y_train == 1) ]

lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)

lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel('Petal length [std]')
plt.ylabel('Petal width [std]')
plt.tight_layout()
plt.legend(loc='upper left')

plt.show()

sys.exit(1)








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


from plot import plot_decision_regions
import matplotlib.pyplot as plt

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
#print(y_combined)
#print(y_combined - np.array(y_train.tolist()+y_test.tolist()))

plot_decision_regions(X = X_combined_std, y = y_combined,
						classifier = ppn,
						test_idx = range(y.shape[0]-y_test.shape[0],y.shape[0]))

plt.xlabel('Petal length [std]')
plt.ylabel('Petal width [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
