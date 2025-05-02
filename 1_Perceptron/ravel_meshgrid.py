import matplotlib.pyplot as plt
import numpy as np

nx, ny = (3,2)

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)

xv, yv = np.meshgrid(x,y)

print(f'xvectors : {xv}')
print(f'yvectors : {yv}')
#plt.plot(xv,yv)

xv = xv.ravel()
yv = yv.ravel()
print(f'raveled - xvectors : {xv}')
print(f'raveled - yvectors : {yv}')
#plt.plot(xv.ravel(),yv.ravel(),linestyle='',marker='o')

xy = np.array([xv,yv]).T
print(xy)
print(xy.shape)	# numpy array.shape !!! # DataFrame.values !!! > note values return numpy array

arr = np.arange(15)
arr = np.arange(12)

arr = arr.reshape(xy.shape) # reshaping 
print(arr)

plt.show()
