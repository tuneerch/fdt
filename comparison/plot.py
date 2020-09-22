import numpy as np
import matplotlib.pyplot as plt

ds = np.loadtxt('data/schur.dat')
de = np.loadtxt('data/evol.dat')
plt.plot(ds[0,:], ds[1,:], 'k.-', label = 'Schur')
plt.plot(de[0,:], de[1,:], 'b.-', label = 'Evolution')

plt.legend()
plt.grid()
plt.xlabel('$\gamma\\tau$')
plt.ylabel('$1-S_\infty$')
plt.savefig('/home/tuneer/Dropbox/project/notes1/figs/comparison_chain.png')
plt.show()
