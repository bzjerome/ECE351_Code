# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:29:16 2020

@author: jero7605
"""

############################################
#                                          #
# Brady Jerome                             #
# ECE351-52                                #
# Lab 4                                    #
# 2/20/2020                                #
# Convolution                              #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
import control as con
import pandas as pd
import time
from scipy.fftpack import fft, fftshift

#----------- Step Function -----------
def step(t):
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            x[i] = 1
        else:
            x[i] = 0
    return x

#----------- Ramp Function -----------
def ramp(t):
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            x[i] = 0
        else:
            x[i] = t[i]
    return x

#%% Plotting graphs  
steps = 1e-2
t = np.arange(0, 10, steps)

def f1(t):
    x = (step(t-2)-step(t-9))
    return x

def f2(t):
    x = np.exp(-t)*step(t)
    return x

def f3(t):
    x = (ramp(t-2)*(step(t-2)-step(t-3))+ramp(4-t)*(step(t-3)-step(t-4)))
    return x

plt.figure(figsize = (5,7))
plt.subplot(3, 1, 1)
plt.plot(t,f1(t))
plt.title('Plotted Functions')
plt.ylabel('f2(t)')
plt.ylim([0, 1.2])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, f2(t))
plt.ylabel('f1(t)')
plt.ylim([0, 1.2])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, f3(t))
plt.ylim([0, 1.2])
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel('t')
plt.show()

#%% Convolution
def conv(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1, np.zeros((1, Nf2-1)))
    f2Extended = np.append(f2, np.zeros((1, Nf1-1)))
    c = np.zeros(f1Extended.shape)
    for i in range(Nf1 + Nf2 - 2):
        c[i] = 0
        for j in range(Nf1):
            if(i-j+1 > 0):
                try:
                    c[i]+= f1Extended[j]*f2Extended[i-j+1]
                except:
                        break;
    return c

steps = 1e-2
t = np.arange(0, 20 + steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN - 1], steps)

f1 = f1(t)
f2 = f2(t)
f3 = f3(t)

#%% Graphing f1 * f2
conv12 = conv(f1, f2)*steps
conv12Check = sig.convolve(f1, f2)*steps


plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv12, label = 'User-Defined Convolution')
plt.plot(tExtended, conv12Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f1(t) * f2(t)')
plt.title('Convolution of f1 and f2')
plt.show()

#%% Graphing f2 * f3

conv23 = conv(f2, f3)*steps
conv23Check = sig.convolve(f2, f3)*steps


plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv23, label = 'User-Defined Convolution')
plt.plot(tExtended, conv23Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f3(t) * f1(t)')
plt.title('Convolution of f2 and f3')
plt.show()

#%% Graphing f1 * f3

conv13 = conv(f1, f3)*steps
conv13Check = sig.convolve(f1, f3)*steps


plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv13, label = 'User-Defined Convolution')
plt.plot(tExtended, conv13Check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f1(t) * f3(t)')
plt.title('Convolution of f1 and f3')
plt.show()