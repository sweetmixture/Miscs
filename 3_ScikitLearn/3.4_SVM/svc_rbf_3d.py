from sklearn.svm import SVC, SVR
from sklearn.datasets import make_circles
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# complicated non-linear data (classification)
X_circ, y_circ = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=42)

# scaling
X_circ_scaled = StandardScaler().fit_transform(X_circ)

# non-linear kernel SVC (RBF)
svc_rbf = SVC(kernel='rbf', C=1.0, gamma='auto')
svc_rbf.fit(X_circ_scaled, y_circ)

# decision function calculation (z-axis)
x0 = np.linspace(X_circ_scaled[:, 0].min(), X_circ_scaled[:, 0].max(), 100)
x1 = np.linspace(X_circ_scaled[:, 1].min(), X_circ_scaled[:, 1].max(), 100)
xx, yy = np.meshgrid(x0, x1)
Z = svc_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 3D visulaisation
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, Z, cmap='coolwarm', alpha=0.7)
ax.scatter(X_circ_scaled[:, 0], X_circ_scaled[:, 1], 
		   svc_rbf.decision_function(X_circ_scaled), 
		   c=y_circ, cmap='coolwarm', edgecolor='k')
ax.set_title("SVC with RBF Kernel - 3D Decision Function")

plt.tight_layout()
plt.show()
