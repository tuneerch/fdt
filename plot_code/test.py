import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

theta = np.linspace(-np.pi, np.pi, 1000)
tau = 10.
def p(t):
    #sig = 0.2
    #ret = np.exp(-t**2/(2*sig**2)) / np.sqrt(2*sig**2*np.pi)
    nf = np.floor(tau/np.pi + 0.5)
    ret = np.zeros(len(t))
    for n in np.arange(-nf,nf+1):
        ret += np.where(np.abs(t+2*n*np.pi) >= 2*tau, 0, 1./(np.pi * np.sqrt(4.*tau**2-(t+2*n*np.pi)**2)))
    return(ret)

#r = 0.99


def k(t,r):
   ret = 2*r*np.sin(t) / (1+r**2-2*r*np.cos(t))
   return(ret)

#def y1(t):
#    ret = integrate.trapz(p(t-theta)*k(theta), theta)
#    return(ret)
#y1_vec = np.vectorize(y1)

#def y(t,r):
#    ret = integrate.trapz(p(theta)*k(t-theta,r), theta)
#    return(ret)
#y_vec = np.vectorize(y)
#y_ex = 1./np.tan(theta/2.)
##plt.plot(theta, k(theta))
##plt.plot(theta, y_vec(theta))
#print(integrate.trapz(p(theta), theta))
#plt.plot(theta, y_ex, 'k', label='$\cot(\\theta/2)$')
#plt.plot(theta, y_vec(theta,0.9), label='r=0.9')
#plt.plot(theta, y_vec(theta,0.99), label='r=0.99')
#plt.plot(theta, y_vec(theta,0.999), label='r=0.999')
#plt.plot(theta, y_vec(theta,0.9999), label='r=0.9999')
#plt.grid()
#plt.xlabel('$\\theta$')
#plt.ylabel('$\Im F(re^{i\\theta})$')
#plt.ylim([-10,10])
#plt.legend()

#plt.savefig('/home/tuneer/Dropbox/project/notes1/figs/convergence')

print(integrate.trapz(p(theta),theta))
plt.plot(theta,p(theta))
plt.show()

#
#
