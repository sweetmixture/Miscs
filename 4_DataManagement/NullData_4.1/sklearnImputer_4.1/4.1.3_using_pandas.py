#
#
#
from sklearn.impute import SimpleImputer # only support columnwise interpolation
import pandas as pd
import numpy as np

df = pd.read_csv('sample.csv',sep=',')

print(f' * numpy isnan()')
for k in df.index.values:
	print(np.isnan(df.iloc[k].values))

print(f' * numpy isnull()')
print(df.isnull())
isnull_arr = df.isnull().values
for k in range(isnull_arr.shape[0]):
	print(isnull_arr[k])	# access numpy array rows

print(f' * ------------------------')

print(f' # pandas fillna : mean')
mean_df = df.fillna(df.mean())
print(mean_df)
print(f' # pandas fillna : bfill - 1')
backfill_df = df.fillna(method='bfill',axis=0) # this will be deprecated
print(backfill_df)
print(f' # pandas fillna : bfill - 2')
backfill_df = df.bfill(axis=0)
print(backfill_df)
print(f' # pandas fillna : bfill - 3 using column')
backfill_df = df.bfill(axis=1)
print(backfill_df)

print(f' # pandas fillna : ffill - 1')
forwardfill_df = df.ffill()
print(forwardfill_df)
print(f' # pandas fillna : ffill - 2')
forwardfill_df = df.ffill(axis=1)
print(forwardfill_df)

