import numpy as np

np.random.seed(12)

rand_source = np.random.randn(20)

print(f'rand list : {rand_source}')

# np.where
binary_arr = np.where(rand_source > 0, 1, 0)	# rand_source is a numpy array

print(f'binary array   : {binary_arr}')
print(f'binary bin cnt : {np.bincount(binary_arr)}')

arr = np.arange(0,binary_arr.shape[0],1)
print(arr)

# selective extraction
arr1 = arr[ binary_arr == 1 ]
arr2 = arr[ binary_arr == 0 ]
arr3 = arr[ binary_arr == 3 ]
arr4 = arr[ [ True for i in range(binary_arr.shape[0]) ] ] 
print(f'arr1 : {arr1}')
print(f'arr2 : {arr2}')
print(f'arr3 : {arr3}')
print(f'arr4 : {arr4}')
print(f' * --------------------')
print(binary_arr == 1)
print(binary_arr == 0)
print(binary_arr == 3)
