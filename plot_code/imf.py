import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

theta = np.linspace(-np.pi,np.pi,1000)
r = 0.999

tau = 3.3
a,b =-0.1,.1

def k(t):
    return(2*r*np.sin(t)/(1+r**2-2*r*np.cos(t)))

def p(t):
    if(t>=a and t<b): ret = 1./(b-a)#/(2.*np.pi)
    else:   ret = 0.
    #return(ret)
    #if(np.abs(t)<tau):  ret = 1./(np.pi * np.sqrt(4.-(float(t)/tau)**2))
    #else:   ret = 0
    return(ret)

#p_vec = np.vectorize(p)
def p_vec(t):
    nf = np.floor(tau/np.pi + 0.5)
    ret = np.zeros(len(t))
    for n in np.arange(-nf,nf+1):
        ret += np.where(np.abs(t+2*n*np.pi) >= 2*tau, 0, 1./(np.pi * np.sqrt(4.*tau**2-(t+2.*n*np.pi)**2)))

    #ret = np.where(np.abs(t) >= 2*tau, 0, 1./(np.pi * np.sqrt(4.*tau**2-t**2)))
    #sig = 0.2
    #ret = np.exp(-(t/sig)**2/2) / np.sqrt(2*np.pi* sig**2)
    return(ret)
k_vec = np.vectorize(k)

'''imaginary part of F'''
def y(t):
    ret = integrate.trapz(p_vec(theta)*k(t-theta), theta)
    return(ret)
y_vec = np.vectorize(y)

imF = y_vec(theta)
reF = 2*np.pi * p_vec(theta)
f2 = ((reF-1.)**2 + imF**2) / ((reF+1.)**2 + imF**2)

#plt.subplot(211)
#plt.title('r='+str(r)+', $\gamma\\tau=$'+str(tau))
#plt.plot(theta, reF, 'b-', label = '$\Re F(re^{i\\theta})$')
#plt.plot(theta, imF, 'r-', label = '$\Im F(re^{i\\theta})$')
#plt.grid()

##plt.xlabel('$\\theta$')
#plt.legend()
##plt.plot(theta, k_vec(theta), 'g-')
#plt.subplot(212)
#plt.plot(theta, f2, 'k-')
#plt.xlabel('$\\theta$')
#plt.ylabel('$|f(re^{i\\theta})|^2$')
#plt.ylim([0,1.5])
#plt.grid()
#
#
##plt.savefig('/home/tuneer/Dropbox/project/notes1/figs/f2.png')
#plt.show()
#R = integrate.trapz(f2, theta)/(2*np.pi)
#print(R)
plt.plot(theta, p_vec(theta), 'k-')
print(integrate.trapz(p_vec(theta),theta))
plt.show()

