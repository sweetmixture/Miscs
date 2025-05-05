import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()
sys.path.append(cwd)

from Adaline import AdalineGD

# load data
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s,header=None,encoding='UTF-8')
# target
y = df.iloc[0:100,4].values # save species : 'Iris-setosa' / 'Iris-versicolor'
y = np.where(y=='Iris-setosa',0,1)  # if 'Iris-setosa' > 1, else > 0 : return numpy array
# features
X = df.iloc[0:100,[0,2]].values # to numpy array # print(X.shape) : numpy.shape

# -----------------------------------------------------------------------------------------

max_iter = 15
eta1 = 0.1
eta2 = 0.0001

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,4)) #fig, ax = plt.subplots(1,2, figsize=(10,4))

ada1 = AdalineGD(n_iter=int(f'{max_iter}'), eta=float(f'{eta1}')).fit(X,y)
ax[0].plot(range(1,len(ada1.losses_)+1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(MSE)')
ax[0].set_title('Adaline - learning rate 0.1')

ada2 = AdalineGD(n_iter=int(f'{max_iter}'), eta=float(f'{eta2}')).fit(X,y)
ax[1].plot(range(1,len(ada2.losses_)+1), np.log(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(MSE)')
ax[1].set_title('Adaline - learning rate 0.0001')

plt.show()
