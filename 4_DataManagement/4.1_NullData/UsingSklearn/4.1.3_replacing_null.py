#
# Using impute module, SimpleImputer class
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

df = pd.read_csv('sample.csv',sep=',' ) # , names = [ ... ] 
for k in df.index.values:
	#print( np.where(df.iloc[k].values == np.nan, True, False) )		# this will always return False since np.nan == np.nan -> False always, Nan != Nan by IEEE 754
	print( np.isnan( df.iloc[k].values ) )
# does the same
print(df.isnull())

arr = df.values
col = df.columns
print(arr)
print(' * back to DataFrame')
#back2df = pd.DataFrame(arr, names=col) # names will not work
back2df = pd.DataFrame(arr, columns=col)
print(back2df)

print('# Impute Mean() -------------------------------------------------------------')
print(df)

imr = SimpleImputer(missing_values=np.nan, strategy='mean') # mean of back/forward values of columns
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
#
# (1) fit > (2) transform
#
print('imputed data\n',imputed_data)
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
print(df_imputed)
