import numpy as np

a = np.array([1,2,3,4,5])
b = [True,True,False,False,True]

a = a[b]
print(a)
# Expected Output
# [1 2 5]

a = np.array(np.arange(1,11,1))

b0 = np.array([0 for i in range(5)])
b1 = np.array([1 for i in range(5)])
b = np.hstack((b0,b1))
np.random.seed(1)
np.random.shuffle(b)

a0 = a[ (b == 0) ]
a1 = a[ (b == 1) ]

print(a0,a1)
