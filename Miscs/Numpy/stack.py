import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])

c = np.vstack((a,b))
print(c)

h = np.hstack((a,b))
print(h)
