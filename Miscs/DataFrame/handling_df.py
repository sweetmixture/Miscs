import numpy as np
import pandas as pd

df = pd.read_csv('sample.dat',sep=',')

print(df)
print(df.iloc[[1]], df.iloc[[1]].values)    # row access return DataFrame
print(df.iloc[1], df.iloc[1].values)        # row access return Series

print(' * reshape & back 2 df')
df2 = pd.DataFrame(df.iloc[1].values.reshape((1,-1)))
print(df2)

for row_index in df.index:
    print(df.iloc[row_index, [ k for k in range(df.columns.values.shape[0])] ] )
