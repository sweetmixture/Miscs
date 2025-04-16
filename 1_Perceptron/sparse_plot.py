import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

s = 'https://archive.ics.uci.edu/ml/'\
	'machine-learning-databases/iris/iris.data'

df = pd.read_csv(s,header=None,encoding='UTF-8')

y = df.iloc[0:100,4].values # values -> return numpy.array : extract row 0:100 / column 4 -> will be used as target later
print('----------------------------------')
print(y,type(y))
print('----------------------------------')
# select only 'Iris-setosa'
y = np.where(y == 'Iris-setosa', 0, 1)
print(y)
print('----------------------------------')
X = df.iloc[0:100, [0,2]].values # <np.ndarray> : extract row 0:100 / column 0 and 2
print(X)

plt.scatter(X[:50,0], X[:50,1], color='r', marker='o', label='Setosa')
plt.scatter(X[50:,0], X[50:,1], color='b', marker='o', label='Versicolor')

plt.legend(loc='upper left')
plt.show()
