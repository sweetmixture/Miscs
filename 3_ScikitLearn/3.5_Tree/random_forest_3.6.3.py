import sys
import matplotlib.pyplot as plt
import numpy as np

from plot import plot_decision_regions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# Random Forest -----------------------------------------------
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
	n_estimators = 100,
	random_state = 1,
	max_depth = 10,
	n_jobs = 4,			# in parallel?
)

forest.fit(X_train,y_train)

X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.xlabel('Petal legnth [cm]')
plt.ylabel('Petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.show()

predict_result = forest.predict(X_combined)
print(predict_result,predict_result.shape)

result = np.where( predict_result == y_combined, 1, 0 )
print(result)

result = np.array( [ 1 if model == target else 0 for model, target in zip(predict_result,y_combined) ] )
print(result)

class_label, counts = np.unique( result, return_counts=True)

print(f' * prediction rate : {counts[1]/counts.sum()*100:6.2f} %')


