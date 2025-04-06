import numpy as np

X = [[1,2,3],[4,5,6],[2,1,0]]
X = np.array(X)

print(X)
print('---------')
for x in X:
	print(x,type(x))
print('---------')
for x in zip(X):
	print(x,type(x))
print('---------')
# expected output
# [1 2 3] <class 'numpy.ndarray'>
# [4 5 6] <class 'numpy.ndarray'>
# [2 1 0] <class 'numpy.ndarray'>
# ---------
# (array([1, 2, 3]),) <class 'tuple'>
# (array([4, 5, 6]),) <class 'tuple'>
# (array([2, 1, 0]),) <class 'tuple'>

w = np.array([1,2,3])
# w (1 x 3) * X (3 x 3) 
r = np.dot(w.T,X)
#r = np.dot(w,X)			# this works too : why?
print(r,r.shape)

print('---------')
r = np.dot(X,w)
print(r,r.shape)
