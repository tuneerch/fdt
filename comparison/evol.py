import numpy as np
import matplotlib.pyplot as plt

L = 601 # chain length is 2L+1
gamma = 1.
#tau = 0.4

x_det = 0 #detector position
x_init = 0 #starting point of particle

x = np.arange(L)

#a = np.zeros((L,L))
#for k in range(-L,L+1):
#    a[k+L] = 1./np.sqrt(L+1) * np.sin(np.pi/2. * (k+L+1.) * (x+L+1.) / (L+1.))

'''defining energy coeff matrix a[k,x] := <E_i|x>'''
a = 1./np.sqrt(L) * np.exp(2J*np.pi/L * np.outer(x,x)) # k's and x's have same vals in this convention

en_arr = -2. * gamma * np.cos(2*np.pi/L*x)
n_max = int(10000)
S = np.zeros(n_max)

'''Fuction for finding the knee point of S curve. Has nothing to do with dynamics'''
def argknee(s):
    f = -np.diff(np.append(1,s))
    for i in range(1,len(f)): #finding the first min of f
        if(f[i-1] > f[i] and f[i+1] > f[i]):
            firstminarg = i
            break
    
    inflecarg = firstminarg + np.argmax(f[firstminarg:]) #arg of nontrivial max of f. also point of inflection of s
    for i in range(1,inflecarg):
        ind = inflecarg-i
        if((f[ind-1] > f[ind]) and (f[ind+1] > f[ind])):
            kneearg = ind
            break
    return(kneearg)

def getR(tau):
    U = np.diag(np.exp(-1J *tau* en_arr))
    U = np.inner(a,U)
    U = np.inner(U,np.transpose(np.conjugate(a)))
    U[x_det,:] = 0 #placing detector at x_det
    
    psi = np.zeros(L)
    psi[x_init] = 1. #initialising state at x_init
    
    for n in range(n_max): #evolution loop
        psi = np.dot(U,psi)
        S[n] = np.real(np.inner(np.conj(psi),psi))
    
    return(1-S[argknee(S)])
    
#F = -np.diff(np.append(1,S))
#plt.subplot(211)
#plt.semilogx(S,'-')
##plt.xlim([0,800])
#plt.subplot(212)
#plt.loglog(F,'.-')
##plt.xlim([0,800])
#n_mean = np.sum(np.arange(1,len(F)+1) * F)
#print(n_mean)
#
#plt.show()

tau_arr = np.linspace(0.1,5,50)
R_arr = np.empty((0))
for tau in tau_arr:
    try:
        R_arr = np.append(R_arr, getR(tau))
        print(tau)
    except:
        print("problem at tau="+str(tau))
        break

data = np.array([tau_arr, R_arr])
np.savetxt("data.dat", data)
plt.plot(tau_arr, R_arr, '.-')
plt.show()
