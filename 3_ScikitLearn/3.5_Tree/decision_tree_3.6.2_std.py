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

# * standardise
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

# Tree model -----------------------------------------------
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion='gini',
	max_depth=4,
	max_features=None,	# default -> using all features
	random_state=1
	)

tree_model.fit(X_train_std,y_train)

#print(X_train.shape,X_test.shape)
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

#a=range(0,10)
#print(a)
plot_decision_regions(X_combined_std, y_combined, classifier=tree_model, test_idx=range(105,150))

plt.xlabel('Petal legnth [std]')
plt.ylabel('Petal width [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(' * predict result')
predict = tree_model.predict(X_combined_std)
# shape check
#print(predict.shape,y_combined.shape)

result = np.array([ True if model == target else False for model, target in zip(predict,y_combined) ])
success_count = np.where(result == True, 1, 0).sum()
total_count = result.shape[0]

# generic method for more than 2 classes
unique, counts = np.unique(result, return_counts=True)
print(unique,counts)
print(dict(zip(unique,counts)))

print(f' * prediction success rate : {success_count/total_count*100:6.2f} %')
for item in counts:
	print(f' * prediction success rate : {item/counts.sum()*100:6.2f} %')

# * tree plot

from sklearn import tree

print(iris.keys())
print(iris['feature_names'])

feature_names = iris['feature_names']

tree.plot_tree(tree_model, feature_names = feature_names, filled=True)

plt.show()


