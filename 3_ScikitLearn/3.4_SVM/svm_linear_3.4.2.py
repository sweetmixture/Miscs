import matplotlib.pyplot as plt
from sklearn.svm import SVC
'''
	SVC : Support Vector Classification

		> classification purpose : find decision boundary by maximising the margin
'''
import numpy as np
from sklearn import datasets
from plot import plot_decision_regions

# data load
iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

# data division
from sklearn.model_selection import train_test_split

random_state = 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y) # stratification

print(f' * result of stratification')
print(f' y       label count : {np.bincount(y)}')
print(f' y train label count : {np.bincount(y_train)}')
print(f' y test  label count : {np.bincount(y_test)}')

# ------------------------------------------------------

# data standardisation
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)

# apply scaling
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

# model fitting
svm = SVC(kernel='linear', C=1.0, random_state=1)
#svm = SVC(kernel='rbf', C=10**12, random_state=1)	# nonlinear kernel 'rbf'
#svm = SVC(kernel='rbf', gamma=0.5 , C=1.0, random_state=1)	# nonlinear kernel 'rbf' # 'gamma' term in the rbf kernel function : large gamma - overfitting
svm.fit(X_train_std,y_train)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined_std,y_combined,
					classifier=svm,
					test_idx=range(np.sum(y_train),np.sum(y)))

plt.legend(loc='upper left')
plt.show()
