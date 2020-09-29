import numpy as np
import matplotlib.pyplot as plt

L = 5
x = np.arange(L)
a = np.sqrt(2./(L+1.)) * np.sin(np.pi * np.outer((x+1.),(x+1.)) / (L+1.))

def s_inf(l,m):
    ret = 0
    for k in range(L):
        if ((k+1)*(m+1)%(L+1)==0):
            ret += np.abs(a[l,k])**2
    return(ret)

def myplot(l,m, tick = True):
    filename = './mean_times_diff/5chain' + str(l) + '_' + str(m) + '.dat'
    data = np.loadtxt(filename, comments = '#')
    #if tick == False:
        #plt.ticks([])
    plt.plot(data[0]/np.pi, data[2], color = 'b', label = str(l+1)+'$\\rightarrow$'+str(m+1))
    plt.plot(data[0]/np.pi, (1-s_inf(l,m))*np.ones(data[0].shape), 'k--', label = str(l+1)+'$\\rightarrow$'+str(m+1) + ' predicted')
    plt.xlabel('$\gamma \\tau /\pi$')
    plt.ylabel('$1-S_{\infty}$')
    plt.ylim([0,1.2])
    plt.legend()
    plt.grid()

    plt.savefig('figs/s_inf/5chain' + str(l) + '_' + str(m)+ '.png')


#def myplot(l,m,o,p, tick = True):
#    filename = './mean_times_diff/5chain' + str(l) + '_' + str(m) + '.dat'
#    data = np.loadtxt(filename, comments = '#')
#    plt.plot(data[0]/np.pi, data[2], color = 'b', label = str(l+1)+'$\\rightarrow$'+str(m+1))
#    plt.plot(data[0]/np.pi, (1-s_inf(l,m))*np.ones(data[0].shape), 'k--', label = str(l+1)+'$\\rightarrow$'+str(m+1) + ' predicted')
# 
#    filename = './mean_times_diff/5chain' + str(o) + '_' + str(p) + '.dat'
#    data = np.loadtxt(filename, comments = '#')
#    plt.plot(data[0]/np.pi, data[2], color = 'y', label = str(o+1)+'$\\rightarrow$'+str(p+1))
#    plt.plot(data[0]/np.pi, (1-s_inf(o,p))*np.ones(data[0].shape), 'r--', label = str(o+1)+'$\\rightarrow$'+str(p+1) + ' predicted')
# 
#    plt.ylim([0,1.2])
#    #if tick == False:
#    #    plt.xticks([])
#    plt.xlabel('$\gamma \\tau /\pi$')
#    plt.ylabel('$1-S_{\infty}$')
#    plt.legend()
#    plt.grid()
#    plt.savefig('./figs/s_inf/5chain' + str(l) + '_' + str(m)+ '.png')


myplot(1,3, False)
plt.show()

