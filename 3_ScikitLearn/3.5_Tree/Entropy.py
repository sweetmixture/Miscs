import numpy as np
import matplotlib.pyplot as plt

def entropy(p):

	return -p * np.log2(p) - (1. - p) * np.log2(1. - p)

#x = np.linspace(0., 1., 100)	# linspace include last element
x = np.arange(0.,1.,0.01)		# arange does not include last element
ent = np.array([ entropy(p) if p != 0 else None for p in x ])

print(x.shape,ent.shape)

plt.plot(x,ent)
plt.xlabel('Class-membership probability p(i=1)')
plt.ylabel('Entropy')
plt.show()
