import numpy as np
import matplotlib.pyplot as plt

ds = np.loadtxt('data/schur.dat')
de = np.loadtxt('data/evol.dat')
db = np.loadtxt('data/bessel.dat')
plt.plot(de[0,:], de[1,:], 'b.-', label = 'Evolution Estimate')
plt.plot(db[0,:], db[1,:], 'k-', label = 'Exact Series Summation')
plt.plot(ds[0,:], ds[1,:], 'r.', label = 'Schur Integration')

plt.legend()
plt.grid()
plt.xlabel('$\gamma\\tau$')
plt.ylabel('$1-S_\infty$')
#plt.savefig('/home/tuneer/Dropbox/project/notes1/figs/comparison_chain.png')
plt.show()
