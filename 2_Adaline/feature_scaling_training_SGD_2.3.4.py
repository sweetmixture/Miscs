import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cwd = os.getcwd()
sys.path.append(cwd)

from Adaline import AdalineSGD
from plot import plot_decision_regions

# load data
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s,header=None,encoding='UTF-8')
# target
y = df.iloc[0:100,4].values # save species : 'Iris-setosa' / 'Iris-versicolor'
y = np.where(y=='Iris-setosa',0,1)  # if 'Iris-setosa' > 1, else > 0 : return numpy array : actually the 'else' is formed of two-species
# features
X = df.iloc[0:100,[0,2]].values # to numpy array # print(X.shape) : numpy.shape

# -----------------------------------------------------------------------------------------

# feature scaling
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada_gd = AdalineSGD(n_iter=20,eta=0.01,random_state=1)
ada_gd.fit(X_std,y)

plot_decision_regions(X_std,y,classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardised]')
plt.ylabel('petal length [standardised]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada_gd.losses_)+1),ada_gd.losses_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.tight_layout()

plt.show()
