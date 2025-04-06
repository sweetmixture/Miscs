import pandas as pd

data = {
    'code'  : range(2000, 2010),
	'score' : [85, 95, 75, 70, 100, 100, 95, 85, 80, 85]
}

print('dataframe load')
df = pd.DataFrame(data)

print(df)
print(df.head(10))

#print(df[:,0])
print(df.shape)

print('------')
print(df.loc[:,['code','score']])
print('------')
print(df.iloc[:,0])
print('------')
print(df.iloc[:,0:1])
print('------')
print(df.iloc[:,-1])
print('------')
dff = df[ df['score'] > 80 ]
print(dff)
print('------')
dff = df[ (df['score'] > 80) & (df['code'] < 2005) ]
print(dff)

print('------')
# this will loop columns
for x in df:
	print(x,type(x))
for x in zip(df):
	print(x)



