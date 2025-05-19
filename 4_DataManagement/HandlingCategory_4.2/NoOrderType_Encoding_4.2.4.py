import pandas as pd
import numpy as np

df = pd.read_csv('sample.csv',sep=',')
print(df)

#
#print(df[:,0])	# numpy based acess not working
print(df.iloc[:,0])
print(df.iloc[:,[0,1,2]])
print(df.iloc[:2,:2])
print(df.loc[:3,['classlabel']])

df2 = df[['color']]
print(df2,type(df2))
#df2 = df[[2]]			# key error : 2 is not in column
#print(df2,type(df2))
#df2 = df[2]			# key error : ...
#print(df2,type(df2))

print(f' iloc[2] ------')
df2 = df.iloc[2]		# access row 2 return Series
print(df2,type(df2))
print(f' iloc[[2]] ------')
df2 = df.iloc[[2]]		# access row 2
print(df2,type(df2))
print(f' iloc[[2,2]] ------')
df2 = df.iloc[[2,2]]		# access row 2 and row 2 twice!!
print(df2,type(df2))
print(f' iloc[[1,2]] ------')
df2 = df.iloc[[1,2]]		# access row 1 and row 2
print(df2,type(df2))
print(f' iloc[[2],[0]] ------')
df2 = df.iloc[[2],[0]]		# access row 2 col 0 > row,col order expected
print(df2,type(df2))

print(f' df["size"] ------')
df2 = df['size']		# access col size return Series
print(df2,type(df2))
print(f' df[["size"]] ------')
df2 = df[['size']]		# access col size return DataFrame
print(df2,type(df2))
print(f' * end of test -----------------------------')


print(f' ----------------------------------')
print(f' using LabelEncoder')
print(f' ----------------------------------')
from sklearn.preprocessing import LabelEncoder
X = df[['color','size','price','classlabel']].values
color_le = LabelEncoder()
y = color_le.fit_transform( X[:,0] )	# fit_transform method expect 1d arr input
print(y)
X[:,0] = y
print(X)
y = color_le.fit_transform( X[:,3] )
X[:,3] = y
print(X)

print(df[0:1])
'''
	DataFrame Access

	df['column label']								# return Series
	df[['column label', 'column label', ... ]]		# return DataFrame

	df.iloc[n]			# return row n Series
	df.iloc[[n,m,...]]	# return row n,m, ... DataFrame

	df.iloc[[n,..],[m,..]] # return row n,.. and col m,.. DataFrame


	DataFrame add a new column

	df['D'] = df['A'] + 5	# create column 'D' using 'A'
'''
