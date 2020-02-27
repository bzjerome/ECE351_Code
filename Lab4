# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:29:58 2020

@author: jero7605
"""

############################################
#                                          #
# Brady Jerome                             #
# ECE351-52                                #
# Lab 4                                    #
# 2/27/2020                                #
# Trasnfer functions                       #
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

#%% Part 1 Task 1
    
# Defining Transfer functions
    
steps = 1e-3
t = np.arange(-10, 10, steps)

f = 0.25
w = (f*(np.pi*2))

def h1(t):
    x = (np.exp(2*t)*step(1-t))
    return x

def h2(t):
    x = (step(t-2) - step(t-6))
    return x
    
def h3(t):
    x = (np.cos(w*t)*step(t))
    return x

#%% Part 1 Task 2

# Plotting the Transfer functions
    
plt.figure(figsize = (5,10))
plt.subplot(3, 1, 1)
plt.plot(t, h1(t))
plt.title('Plotted Step Response Functions')
plt.ylabel('h1(t)')
plt.ylim([0, 8])
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, h2(t))
plt.ylabel('h2(t)')
plt.ylim([0, 1.2])
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, h3(t))
plt.ylim([0, 1.2])
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t')
plt.show()

#%% Defining the Convolution equation

# Convolution
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
t = np.arange(-10, 10 + steps, steps)
NN = len(t)
tExtended = np.arange(2*t[0], 2*t[NN - 1] + steps, steps)

h1 = h1(t)
h2 = h2(t)
h3 = h3(t)

#%% Part 2 Task 1

#%% Convolving h1 with the step function

conv1s = conv(h1, step(t))*steps
handcalc1 = (((1/2)*np.exp(2*tExtended)*step(1-tExtended))+((1/2)*np.exp(2)*step(tExtended-1)))


plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv1s, label = 'h1 Step Response')
plt.plot(tExtended, handcalc1, '--', label = 'Hand calculated Step Response')
plt.ylim([0, 4])
plt.xlim([-10, 10])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h1(t) * u(t)')
plt.title('Convolution of h1(t) and u(t)')
plt.show()

#%% Convolving h2 with the step function

conv2s = conv(h2, step(t))*steps
handcalc2 = ((tExtended-2)*step(tExtended-2) - (tExtended-6)*step(tExtended-6))

plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv2s, label = 'h2 Step Response')
plt.plot(tExtended, handcalc2, '--', label = 'Hand Calculated Step Response')
plt.ylim([0, 5])
plt.xlim([-10, 10])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h2(t) * u(t)')
plt.title('Convolution of h2(t) and u(t)')
plt.show()

#%% Convolving h3 with the step function

conv3s = conv(h3, step(t))*steps
handcalc3 = ((2/np.pi)*np.sin(0.5*np.pi*tExtended)*step(tExtended))

plt.figure(figsize = (10, 7))
plt.plot(tExtended, conv3s, label = 'h3 Step Response')
plt.plot(tExtended, handcalc3, '--', label = 'Hand Calculated Step Response')
plt.ylim([0, 1])
plt.xlim([-10, 10])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('h3(t) * u(t)')
plt.title('Convolution of h3(t) and u(t)')
plt.show()
