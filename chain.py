import numpy as np
import matplotlib.pyplot as plt

L = 101 # chain length is 2L+1
gamma = 1.
tau = 0.1

x_det = 50 #detector position
x_init = 50 #starting point of particle

x = np.arange(L)

a = np.zeros((L,L))
#for k in range(-L,L+1):
#    a[k+L] = 1./np.sqrt(L+1) * np.sin(np.pi/2. * (k+L+1.) * (x+L+1.) / (L+1.))

'''defining energy coeff matrix a[x,k] := <x|E_k>'''
a = np.sqrt(2./(L+1.)) * np.sin(np.pi * np.outer((x+1.),(x+1.)) / (L+1.)) # k's and x's have same vals in this convention
en_arr = -2. * gamma * np.cos(np.pi*(x+1.)/(L+1.))

U = np.diag(np.exp(-1J *tau* en_arr))
U = np.inner(a,U)
U = np.inner(U,a)
U[x_det,:] = 0 #placing detector at x_det

n_max = int(1e7)
S = np.zeros(n_max)
psi = np.zeros(L)
psi[x_init] = 1. #initialising state at x_init

for n in range(n_max): #evolution loop
    psi = np.dot(U,psi)
    S[n] = np.real(np.inner(np.conj(psi),psi))
    
F = -np.diff(np.append(1,S))
plt.subplot(211)
plt.loglog(S,'-')
plt.subplot(212)
plt.loglog(F,'-')
n_mean = np.sum(np.arange(1,len(F)+1) * F)
print n_mean

plt.show()
