import pandas as pd
import numpy as np

np.random.seed(12)

randarr = np.random.randn(20)
print(randarr)
barr = np.where( randarr >= 0, 1, 0 )
print(barr)
ubarr, cnt = np.unique(barr, return_counts=True)
print(ubarr, cnt)

print(f' * ---------------------------------------')
print(f'   class mapping')
print(f'   > using df["target label"].map( map-dict )')
print(f' * ---------------------------------------')

df = pd.read_csv('sample.csv', sep=',')
print(df)

#
# define mapping table
# 
#print(np.unique(df['classlabel'].values))
class_mapping = { k : v for v, k in enumerate(np.unique(df['classlabel'].values)) }
print(class_mapping) 
inv_class_mapping = { v : k for v, k in enumerate(np.unique(df['classlabel'].values)) }
print(inv_class_mapping)

print(f' * forward class mapping -------------------')
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
print(f' * backward class mapping ------------------')
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

print(f' ---------------------------------------')
print(f' * using sklearn LabelEncoder')
print(f' > sklearn.preprocessing.LabelEncoder')
print(f' ---------------------------------------')

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('sample.csv', sep=',')
class_le = LabelEncoder()
y = class_le.fit(df['classlabel'].values)
y = class_le.transform(df['classlabel'].values)
print(y)
print(' * before encoding -------------')
print(df)
df['classlabel'] = y	# encoding
print(' * after encoding --------------')
print(df)

inv_y = class_le.inverse_transform(y)
df['classlabel'] = inv_y
print(' * inverse encoding ------------')
print(df)
