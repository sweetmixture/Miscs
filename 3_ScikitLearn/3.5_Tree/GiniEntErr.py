import matplotlib.pyplot as plt
import numpy as np

def gini(p):
	return p*(1 - p) + (1 - p)*(1 - (1-p))	# for a binary tree only

def entropy(p):
	return -p * np.log2(p) - (1 - p) * np.log2((1-p))

def error(p):
	return 1 - np.max([p, 1 - p])

x = np.arange(0,1.0,0.01)

ent = np.array([entropy(p) if p != 0 else 10.**-10 for p in x])
sc_ent = 0.5 * ent
err = np.array([ error(p) for p in x ])
gi = np.array(gini(x))
print(sc_ent.shape,err.shape,gi.shape)


#fig, axs = plt.subplots(2, 2, figsize=(10, 6))  # 2x2 grid
#axs[0, 0].plot(x, y)
#axs[1, 1].plot(x, np.cos(x))

fig, ax = plt.subplots()

for i, lab, ls, c, in zip([ent, sc_ent, gi, err],
						['Entropy','Scaled Entropy', 'Gini impurity', 'Misclassification error'],
						['-','-','--','-.'],
						['black','lightgray','red','green']
						):
	line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=4, fancybox=True, shadow=False)
ax.axhline(y=0.5, lw=1, color='k', linestyle='--')
ax.axhline(y=1.0, lw=1, color='k', linestyle='--')

#plt.ylim([0,1.1])
#plt.xlabel('p(i=1)')
#plt.ylabel('impurity index')

#ax.set(title='Title', xlabel='X Label', ylabel='Y Label')
ax.set_ylim([0,1.1])
ax.set_xlabel('p(i=1)')
ax.set_ylabel('impurity index')

plt.show()
