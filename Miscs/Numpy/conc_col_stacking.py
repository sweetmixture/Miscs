import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c1 = np.c_[a, b]
print(c1)

# equivalent 
c2 = np.vstack((a,b)).T
print(c2)

# column stacking

c3 = np.column_stack((a,b))
print(c3)
