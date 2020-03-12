# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:31:33 2020

@author: jero7605
"""

############################################
#                                          #
# Brady Jerome                             #
# ECE351-52                                #
# Lab Number                               #
# Due Date                                 #
# Any other information                    #
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

#%% Defining Sine method
steps = 1e-5
t = np.arange(0, 1.2e-3 + steps, steps)

R = 1000
L = 27e-3
C = 100e-9

num = [0, (1/(R*C)), 0]
den = [1, (1/(R*C)), (1/(L*C))]

def sine_method():  # Works for any single RLC circuit
    a = -1/(2*R*C)
    w = (1/2)*np.sqrt((1/(R*C)**2)-4*(1/(np.sqrt(L*C)))**2+0*1j)
    p = a + w
    g = 1/(R*C)*p
    g_mag = np.abs(g)
    g_rad = np.angle(g)
    g_degrees = g_rad*180/np.pi
    y = (g_mag/np.abs(w))*np.exp(a*t)*np.sin(np.abs(w)*t + g_rad)*step(t)
    return y

tout, yout = sig.impulse((num, den), T = t)

s = sine_method()

#%% Part 1 Task 1 and Task 2

# Plotting the two sine methods
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, s)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('steps')
plt.title('Created Sine Method')
plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 2)
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('steps')
plt.title('Python sig.impulse()')

#%% Part 2 Task 1

# Step Response
tout, yout = sig.step((num, den), T = t)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('steps')
plt.title('Python Step Response')