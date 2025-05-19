import pandas as pd
import numpy as np
from io import StringIO

tarfile = 'sample.csv'

df = pd.read_csv(
	tarfile,
	sep=',',				# instead of 'delimieter' -> use 'sep' : actually delimiter or sep is alias of the other
	#header=None,			# say there is no header row
	#index_col=False,		# say specific column is index column could be name or column
	#index_col=3,
	#names = [ sequence of hashable items ],
	)
'''
* index=True/False
df.to_csv( ...
	index=False,	# do not write the index column 
	...
	)
'''
print(df)

print(' * -----------------------------------------')
csv_data = \
'''
1, 2, 3, 4
5, 6,, 8
10, 11, 12,
'''
df = pd.read_csv(StringIO(csv_data),names=['A','B','C','D']) # names = [something sequence of hashable] ; hashable - immutable
print(df)

csv_data = \
'''
A, B, C, D
1, 2, 3, 4
5, 6,, 8
10, 11, 12,
'''
#
# Caution
# 4, 6, , 8 > the third ' ' is treated as element - a space

df = pd.read_csv(StringIO(csv_data))
print(df)
print('Access "A"')
print(df['A']) # access A
#print(df[:,'A'])	# cannot use numpy like access?
arr = df.values
print(arr)
keys = df.keys()
for key in keys:
	print(key)
for key in keys:
	subset = df[key]
	print(subset.values)
#
# slice rows of col 'A'
print('slice rows of col ''A''')
df2 = df.loc[:,'A']
print(df2)

# internal conditional slicing
print('slice df elements > 2')
df2 = df[ df[df.columns] > 2 ]
print(df2)

print('slice df col ''A'' > 2')
df2 = df.loc[:,['A']] > 2 	# true false problem
print(df2)

print('sorting ''A''')
df2 = df.sort_values(by='A', ascending=False)
print(df2)
print('sorting ''A'' axis = 0(row) 1(col)')
df2 = df.sort_values(by=0, ascending=False, axis=1) 	# by=(row index)
print(df2)

#
# isnull() method !
# isnull().sum() method

# return True/False to show nullified data
nullified = df.isnull()
print(nullified)
nullified_count = nullified.sum()
print(nullified_count)
print(nullified.mean())

#
# using DataFrame attribute : 'values' -> return numpy-array
#
arr = df.values
print(arr) # NaN : numpy nan

arr2 = arr[:,[2,3]]
print(arr2.T)
print(arr2.T.reshape(3,2))

arr3 = arr2.ravel()
print(arr3.reshape(3,2)) # shape preserved # reshaping -> order base

print(arr3)
print(arr3.shape)
arr3 = arr3.reshape(1,-1)
print(arr3)
print(arr3.shape)

arr4 = np.vstack((arr3,np.array(np.arange(0,arr3.shape[1],1))))
print(arr4)
