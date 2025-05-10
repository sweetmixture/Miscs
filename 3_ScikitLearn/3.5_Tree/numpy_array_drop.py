import numpy as np
import matplotlib.pyplot as plt

# manipulating numpy arrays

arr1d = np.arange(0,10,1)

arr1d_1 = np.delete(arr1d,0)	# drop 1st element
print(arr1d,arr1d_1)			# arr1d intact
arr1d_1 = np.delete(arr1d,2)	# drop 3rd element
print(arr1d,arr1d_1)			# arr1d intact

# 2d array case

x = np.arange(0,4,1)
y = np.arange(0,12,2)

print(' * mesh grid xx ,yy ')
xx, yy = np.meshgrid(x,y)
print(xx)
print(yy)
grid_space = np.array([ xx.ravel(), yy.ravel() ]).T

plt.plot( grid_space[:,0], grid_space[:,1], marker='o', linestyle='')
plt.show()

# drop elements

#grid_space = np.delete(grid_space,-1)	# error : this syntax works only for 1d
grid_space_new = np.delete(grid_space,-1,axis=0) # axis=0 refers to row
print(grid_space_new)

plt.plot( grid_space_new[:,0], grid_space_new[:,1], marker='o', linestyle='')
plt.show()

grid_space_new = np.delete(grid_space,-1,axis=1) # axis=1 refers to column
print(grid_space_new)
