import numpy as np
import matplotlib.pyplot as plt

def myplot(l,m, tick = True):
    filename = './mean_times_diff/5chain' + str(l) + '_' + str(m) + '.dat'
    data = np.loadtxt(filename, comments = '#')
    #if tick == False:
        #plt.ticks([])
    plt.plot(data[0]/np.pi, data[1]/data[2], color = 'k', label = str(l+1)+'$\\rightarrow$'+str(m+1))
    plt.xlabel('$\gamma \\tau /\pi$')
    plt.ylabel('$\langle n \\rangle$')
    plt.ylim([0,20])
    plt.legend()
    plt.grid()
    plt.savefig('figs/meantime/5chain' + str(l) + '_' + str(m)+ '.png')


#def myplot(l,m,o,p, tick = True):
#    filename = './mean_times_diff/5chain' + str(l) + '_' + str(m) + '.dat'
#    data = np.loadtxt(filename, comments = '#')
#    plt.plot(data[0]/np.pi, data[1]/data[2], color = 'b', label = str(l+1)+'$\\rightarrow$'+str(m+1))
#    filename = './mean_times_diff/5chain' + str(o) + '_' + str(p) + '.dat'
#    data = np.loadtxt(filename, comments = '#')
#    plt.plot(data[0]/np.pi, data[1]/data[2], 'k--', label = str(o+1)+'$\\rightarrow$'+str(p+1))
#    plt.ylim([0,20])
#    #if tick == False:
#    #    plt.xticks([])
#    plt.xlabel('$\gamma \\tau /\pi$')
#    plt.ylabel('$\langle n \\rangle$')
#    plt.legend()
#    plt.grid()
#    plt.savefig('./figs/meantime/5chain' + str(l) + '_' + str(m)+ '.png')

myplot(0,4, False)
plt.show()

