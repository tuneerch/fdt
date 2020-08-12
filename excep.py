'''plotting mean arrival time vs gamma*tau'''

import numpy as np
import matplotlib.pyplot as plt
import timing

clock1 = timing.stopclock()

L = 5 # chain length is 2L+1
tau = 0.1

x_init = 1
x_det = 4 
#x_det = int((L-1)/2) #detector position
#x_init = x_det #starting point of particle

x = np.arange(L)

a = np.zeros((L,L))
#for k in range(-L,L+1):
#    a[k+L] = 1./np.sqrt(L+1) * np.sin(np.pi/2. * (k+L+1.) * (x+L+1.) / (L+1.))

'''defining energy coeff matrix a[k,x] := <E_k|x>'''

#ring dynamics
#a = 1./np.sqrt(L) * np.exp(2J*np.pi/L * np.outer(x,x)) # k's and x's have same vals in this convention
#en_arr = -2.*np.cos(2.*np.pi/L*x)

clock1.lap()
#chain dynamics
a = np.sqrt(2./(L+1.)) * np.sin(np.pi * np.outer((x+1.),(x+1.)) / (L+1.)) # k's and x's have same vals in this convention
en_arr = -2. * np.cos(np.pi*(x+1.)/(L+1.))
clock1.lap('coeff matrix creation')

n_max = int(500)
S = np.zeros(n_max)
tau_arr = np.linspace(0,2*np.pi,500)
n_mean_arr = np.ndarray(tau_arr.shape)

for i in range(len(tau_arr)):
    tau = tau_arr[i]
    U = np.diag(np.exp(-1J *tau* en_arr))
    U = np.inner(a,U)
    U = np.inner(U,np.transpose(np.conjugate(a)))
    U[x_det,:] = 0 #placing detector at x_det
    psi = np.zeros(L)
    psi[x_init] = 1. #initialising state at x_init

    for n in range(n_max): #evolution loop
        psi = np.dot(U,psi)
        S[n] = np.real(np.inner(np.conj(psi),psi))
    
    F = -np.diff(np.append(1,S))
    n_mean_arr[i] = np.sum(np.arange(1,len(F)+1) * F)

clock1.lap('finished computation')    
data = np.array([tau_arr,n_mean_arr])
filename = 'test_data/'+str(L)+'chain.dat'
f = open(filename, 'w')
f.write('#L = %d\n'%L)
f.write('#transition: %d -> %d \n'%(x_init,x_det))
f.write('#tau = %f\n'%tau)
f.write('#nmax = %d\n'%n_max)
np.savetxt(f, data)
f.close()
clock1.lap('finished recording')
#plt.plot(tau_arr/np.pi, n_mean_arr)
#plt.grid()
#plt.show()
