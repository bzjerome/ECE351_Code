# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:59:55 2020

@author: bzjer
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import control as con
import pandas as pd
import time
from scipy.fftpack import fft, fftshift

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patches    
    
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = np.array(b)/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = np.array(a)/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'o', ms=10,label='Zeros')
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0)

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'x', ms=10,label='Poles')
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0)

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.legend()

    # set the ticks
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    return z, p, k

#%% Part 1 Task 3- Verying Partial Fraction Expanision

numz = [2, -40]
denz = [1, -10, 16]

[r, p, _] = sig.residuez(numz, denz)
print('r = ', r, '\np = ', p)

#%% Part 1 Task 4- Obtaining the pole-zero plot for H(z)

plt.figure(figsize = (10,7))
[z, p, _] = zplane(numz, denz)
print('z = ', z, '\np = ', p)

#%% Part 1 Task 5- Plotting magnitude and phase response of H(z)

w, h = sig.freqz(numz, denz)

fig = plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.plot(w/np.pi, 20*np.log10(np.abs(h)))
plt.ylabel('Magnitude [dB]')
plt.title('Magnitude and Phase Responses of H(z)')
plt.grid()
plt.subplot(2,1,2)
plt.plot(w/np.pi, np.angle(h))
plt.xlabel('Frequency')
plt.ylabel('Phase [rad]')
plt.grid()
plt.show()
