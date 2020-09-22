import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

theta = np.linspace(-np.pi,np.pi,1000)
r = 0.999

#tau = 2.5
#a,b =-0.1,.1

def k(t):
    return(2*r*np.sin(t)/(1+r**2-2*r*np.cos(t)))
k_vec = np.vectorize(k)

#def p(t):
#    if(t>=a and t<b): ret = 1./(b-a)#/(2.*np.pi)
#    else:   ret = 0.
#    #return(ret)
#    #if(np.abs(t)<tau):  ret = 1./(np.pi * np.sqrt(4.-(float(t)/tau)**2))
#    #else:   ret = 0
#    return(ret)

def p_vec(t, tau):
    nf = np.ceil(2.*tau/np.pi)-1
    ret = np.zeros(len(t))
    for n in np.arange(-nf,nf+1):
        ret += np.where(np.abs(t+2*n*np.pi) >= 2*tau, 0, 1./(np.pi * np.sqrt    (4.*tau**2-(t+2*n*np.pi)**2)))
    return(ret)

def y(t,tau):
    ret = integrate.trapz(p_vec(theta, tau)*k(t-theta), theta)
    return(ret)
y_vec = np.vectorize(y)


def getR(tau):
    '''imaginary part of F'''
    imF = y_vec(theta,tau)
    reF = 2*np.pi * p_vec(theta,tau)
    f2 = ((reF-1.)**2 + imF**2) / ((reF+1.)**2 + imF**2)
    return(integrate.trapz(f2, theta)/(2*np.pi))

tau_arr = np.linspace(0.1,5,50)
R_arr = np.ndarray((0))
for tau in tau_arr:
   R_arr = np.append(R_arr, getR(tau))
   print(tau)

data = np.array([tau_arr, R_arr])
np.savetxt("data/shur.dat", data)
plt.plot(tau_arr, R_arr, '.-')
plt.show()


