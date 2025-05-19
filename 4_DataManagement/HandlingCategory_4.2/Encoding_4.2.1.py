import pandas as pd


column_names = ['color','size','price','classlabel']
df = pd.DataFrame(
	[
		['green','M',10.1,'class2'],
		['red','L',13.5,'class1'],
		['blue','XL',15.3,'class2'],
	]
, columns = column_names
)
# or df.columns = column_names
print(df)

#
# say there is a relation : XL = L + 1 = M + 2
#
size_mapping = {
	'XL' : 3,
	'L'  : 2,
	'M'  : 1,
	'S'  : 0.5,
	'XS' : 0.25,
}

print(' * mapping size')
df['size'] = df['size'].map(size_mapping) # df['target col'].map( dict )
print(df)
#
# using df._append(df) to vstack like function
#
#df2 = df._append(df)
#print(df2.values)
#df2 = pd.DataFrame(df2.values, columns = column_names)
#print(df2)

#
# decoding
#
#for k, v in size_mapping.items(): #for k, v in zip(size_mapping.keys(), size_mapping.values()):
#	print(k,v)
inv_size_mapping = { v : k for k,v in size_mapping.items() }
print(inv_size_mapping)

df_org = df.copy()
#df_org = df			# linked like pointer
df_org['size'] = df['size'].map(inv_size_mapping)
print(df_org)
print(df)

df.to_csv('sample.csv',sep=',', index=False)
