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

'''
	📌 1. Column Access
	Task	Syntax
	Single column				df['A'] or df.A
	Multiple columns			df[['A', 'B']]
	Add a new column			df['D'] = df['A'] + 5
	Drop a column				df.drop('C', axis=1)
	
	📌 2. Row Access
	Task	Syntax
	Row by label				df.loc['row2']
	Row by position				df.iloc[1]
	Slice rows (by position)	df[0:2]					# note. df[0] is not going to work, only for column
	Slice rows (by label)		df.loc['row1':'row2']
	Add new rows				df + df

	📌 3. Cell Access (Row + Column)
	Task	Syntax
	By label					df.loc['row2', 'B']
	By position					df.iloc[1, 1]
	Multiple cells				df.loc[['row1', 'row3'], ['A', 'C']]


	
	4. Filtering rows

	df[df['A'] > 15]          # Filter rows where A > 15
	df[(df['A'] > 10) & (df['B'] < 300)]  # Combine conditions

	5. Apply Functions

	df['A'].apply(lambda x: x * 2)    # Apply to a column
	df.apply(np.sum, axis=0)         # Sum columns
	df.apply(np.sum, axis=1)         # Sum rows

	6. Missing Data Handling

	df.isnull()            # Check for NaNs
	df.fillna(0)           # Fill NaNs				df.bfill(method=...) df.ffill(method=...)
	df.dropna()            # Drop rows with NaNs

	📌 7. General Info
	Task	Syntax
	Shape (rows, cols)	df.shape
	Data types			df.dtypes
	Column names		df.columns
	Row index			df.index
	Quick summary		df.info()
	Descriptive stats	df.describe()

	To Numpy Array		df.values
'''

import pandas as pd
import numpy as np

df = pd.DataFrame({
	'A': [10, 20, 30],
	'B': [100, 200, 300],
	'C': ['x', 'y', 'z']
}, index=['row1', 'row2', 'row3'])

print(df)

# add a new column 'D'
df['D'] = df['A']
print(df)
