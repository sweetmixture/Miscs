import numpy as np

npr = np.array([1,2,3])
print(npr,npr.shape)

npr = npr.reshape(1,-1)
print(npr,npr.shape)

npr = np.array([[1,2,3],[4,5,6]])
print(npr,npr.shape)
# [[1 2 3]
#  [4 5 6]] (2, 3)

npr = npr.reshape(6) # reduce dimension
print(npr,npr.shape)
# [1 2 3 4 5 6] (6,)

npr = npr.reshape(1,-1) # add dimension
print(npr,npr.shape)
# [[1 2 3 4 5 6]] (1, 6)
