import pandas as pd
import numpy as np

df =  pd.read_csv('sample.csv',sep=',')

print(' * org data --------')
print(df)
print(df.keys())
for key in df.keys():
	print(key)
print(df.shape)
print(df.index)
for k in df.index:
	print(k)

print(' * dropping rows including NaN')
df2 = df.dropna(axis=0)
print(df2)
print(df2.values)
print(df2.values.shape)

print(' * dropping cols including NaN')
df2 = df.dropna(axis=1)
print(df2)
print(df2.values)
print(df2.values.shape)
print(df2.values.ravel())
print(df2.values.ravel().reshape(df2.values.shape))

'''
	dropna(how='all')
	dropna(thresh=4)			# if there are non-NaN count less than 4

	dropna(axis=0, subset=['colname']) 			# do dropna (row) for the colname 	# defulat axis=0
	dropna(axis=1, subset=['rowname' or index])	# do dropna (col) for the rowname or row index

	# > axis=0 w.r.t. row
	# > axis=1 w.r.t. col
'''

print(' * renew --------')
print(df)
print(df.shape)
print(df.columns)
print(df.index)
#print(df.rows) # wrong !
print(' * isnull() > returns True/False tells NaN')
print(df.isnull())
print(df.isnull().count())

df2 = df.dropna(subset=['C','D'])
print(df2)
df2 = df.dropna(subset=[ k for k in df.index ], axis=1) # for all rows
df2 = df.dropna(subset=df.index, axis=1) # for all rows # this is the same
print(df2)

for col in df.columns:
	for idx in df.index:
		print( np.isnan(df.loc[idx,col]), np.nan == df.loc[idx,col] ) # np.nan == X not working !

for i in range(df.shape[0]):
	for j in range(df.shape[1]):
		print(df.iloc[i,j], np.isnan(df.iloc[i,j]))

