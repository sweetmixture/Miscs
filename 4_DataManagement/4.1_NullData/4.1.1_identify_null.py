import pandas as pd
from io import StringIO

tarfile = 'sample.csv'

df = pd.read_csv(
	tarfile,
	sep=',',				# instead of 'delimieter' -> use 'sep' : actually delimiter or sep is alias of the other
	#header=None,			# say there is no header row
	#index_col=False,		# say specific column is index column could be name or column
	#index_col=3,
	)
'''
* index=True/False
df.to_csv( ...
	index=False,	# do not write the index column 
	...
	)
'''
print(df)

print(' * -----------------------------------------')

csv_data = \
'''
A, B, C, D
1, 2, 3, 4
5, 6,, 8
10, 11, 12,
'''
#
# Caution
# 4, 6, , 8 > the third ' ' is treated as element - a space

df = pd.read_csv(StringIO(csv_data))
print(df)

#
# isnull() method !
#

# return True/False to show nullified data
nullified = df.isnull()
print(nullified)
nullified_count = nullified.sum()
print(nullified_count)
