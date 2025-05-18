import numpy as np
import pandas as pd

df = pd.read_csv('sample.dat',sep=',')

print(df)
print(df.iloc[[1]], df.iloc[[1]].values)    # row access return DataFrame
print(type(df.iloc[[1]]))
print(df.iloc[1], df.iloc[1].values)        # row access return Series
print(type(df.iloc[1]))

print(' * reshape & back 2 df')
df2 = pd.DataFrame(df.iloc[1].values.reshape((1,-1)))	# reshaping to 2d
print(df2)

for row_index in df.index:
    print(df.iloc[row_index, [ k for k in range(df.columns.values.shape[0])] ] )

print(f' * -------------------------------------------')

r, c = df.shape
print(r,c)

ravelled_df = df.values.ravel()
df2 = pd.DataFrame(ravelled_df.reshape(r,c))
print(df2)
print(df)
