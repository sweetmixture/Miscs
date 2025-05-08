import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

# Classification
X_cls, y_cls = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, random_state=42)

# Regression
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# scaling
scaler_cls = StandardScaler()
X_cls_scaled = scaler_cls.fit_transform(X_cls) # fit-transform together

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# svc fitting
svc = SVC(kernel='linear', C=1.0)
svc.fit(X_cls_scaled, y_cls)

svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr.fit(X_reg_scaled, y_reg)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# --- SVC
x0 = np.linspace(X_cls_scaled[:, 0].min(), X_cls_scaled[:, 0].max(), 100)
x1 = np.linspace(X_cls_scaled[:, 1].min(), X_cls_scaled[:, 1].max(), 100)
xx, yy = np.meshgrid(x0, x1)
#Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()]) # stacks xx and yy as columns side by side
Z = svc.decision_function(np.vstack((xx.ravel(), yy.ravel())).T)
# Z returned as 1d-array
print(Z.shape,xx.shape)
Z = Z.reshape(xx.shape)

ax[0].contourf(xx, yy, Z > 0, alpha=0.3)
ax[0].scatter(X_cls_scaled[:, 0], X_cls_scaled[:, 1], c=y_cls, edgecolors='k')
ax[0].set_title("SVC")

# --- SVR
X_plot = np.sort(X_reg_scaled, axis=0)
y_plot = svr.predict(X_plot)

ax[1].scatter(X_reg_scaled, y_reg, color='gray', label='data')
ax[1].plot(X_plot, y_plot, color='red', label='SVR prediction')
ax[1].set_title("SVR")
ax[1].legend()

plt.tight_layout()
plt.show()

