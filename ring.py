import numpy as np
import matplotlib.pyplot as plt

L = 101 # chain length is 2L+1
gamma = 1.
tau = 5.

x_det = 0 #detector position
x_init = 0 #starting point of particle

x = np.arange(L)

a = np.zeros((L,L))
#for k in range(-L,L+1):
#    a[k+L] = 1./np.sqrt(L+1) * np.sin(np.pi/2. * (k+L+1.) * (x+L+1.) / (L+1.))

'''defining energy coeff matrix a[k,x] := <E_i|x>'''
a = 1./np.sqrt(L) * np.exp(2J*np.pi/L * np.outer(x,x)) # k's and x's have same vals in this convention

en_arr = -2. * gamma * np.cos(2*np.pi/L*x)
U = np.diag(np.exp(-1J *tau* en_arr))
U = np.inner(a,U)
U = np.inner(U,np.transpose(np.conjugate(a)))
U[x_det,:] = 0 #placing detector at x_det

n_max = int(10000)
S = np.zeros(n_max)
psi = np.zeros(L)
psi[x_init] = 1. #initialising state at x_init

for n in range(n_max): #evolution loop
    psi = np.dot(U,psi)
    S[n] = np.real(np.inner(np.conj(psi),psi))
    
F = -np.diff(np.append(1,S))
plt.subplot(211)
plt.semilogx(S,'-')
#plt.xlim([0,800])
plt.subplot(212)
plt.loglog(F,'-')
#plt.xlim([0,800])
n_mean = np.sum(np.arange(1,len(F)+1) * F)
print(n_mean)

plt.show()
