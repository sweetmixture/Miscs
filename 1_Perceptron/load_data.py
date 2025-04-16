import os
import pandas as pd

s = 'https://archive.ics.uci.edu/ml/'\
	'machine-learning-databases/iris/iris.data'
print(f'URL: {s}')

df = pd.read_csv(s,header=None,encoding='UTF-8')
print(df)
print(df.shape)
print(df.columns)
print(df[0])	# access column
print(df.loc[:,0:2])	# column / row must be numeric
print(df.iloc[:,-1])
print('----------------------')
