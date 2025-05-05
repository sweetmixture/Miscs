import numpy as np

rgen = np.random.RandomState(42)

p = rgen.permutation(10)
print(p)
