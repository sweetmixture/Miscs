#
# 3.5.2 Kernel method 'rbf' : radial basis function - Gaussian kernel
#
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

X_xor = np.random.randn(500,2) # standard normal distribution : *args : dimension
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
y_xor = np.where(y_xor,1,0)

plt.scatter(X_xor[ y_xor == 1, 0], X_xor[ y_xor == 1, 1 ], color='blue', marker='s', label='Class 1')
plt.scatter(X_xor[ y_xor == 0, 0], X_xor[ y_xor == 0, 1 ], color='red',  marker='o', label='Class 2')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

plt.tight_layout()
plt.legend(loc='upper left')

from sklearn.svm import SVC
from plot import plot_decision_regions

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=10.0)
svm.fit(X_xor,y_xor)

plot_decision_regions(X_xor,y_xor,classifier=svm)

plt.show()
