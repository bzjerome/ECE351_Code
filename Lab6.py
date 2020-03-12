# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 08:30:46 2020

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

#%% Step Function
def step(t):
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] >= 0:
            x[i] = 1
        else:
            x[i] = 0
    return x

#%% Part 1 Task 1

steps = 1e-5
t = np.arange(0, 2 + steps, steps)

y = ((1/2) + np.exp(-6*t) - (1/2)*np.exp(-4*t))*step(t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.title('Ploting the function y(t)')

#%% Part 1 Task 2

num = [1, 6, 12]
den = [1, 10, 24]

tout, yout = sig.step((num,den), T=t)

plt.figure(figsize = (10, 7))
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.title('Python sig.step')

#%% Part 1 Task 3

num = [1, 6, 12]
den = [1, 10, 24, 0]

[R, P, _]=sig.residue(num, den)
print("R1 =", R, "\nP1 = ", P)

#%% Part 2 Task 1

num = [25250]
den = [1, 18, 218, 2036, 9085, 25250, 0]

[R, P, _]=sig.residue(num, den)
print("\nR2 =", R, "\nP2 = ", P)

#%% Part 2 Task 2

def cos2_method(R, P, t):
    y = 0
    for i in range(len(R)):
        alpha = np.real(P[i])
        omega = np.imag(P[i])
        kmag = np.abs(R[i])
        krad = np.angle(R[i])
        y += kmag*np.exp(alpha*t)*np.cos(omega*t+krad)*step(t)
    return y
 
t = np.arange(0, 4.5 + steps, steps)
 
y = cos2_method(R, P, t)

plt.figure(figsize = (10, 7))
plt.plot(t, y)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('x(t)')
plt.title('Part 2 Task 2')

#%% Part 2 Task 3

num = [25250]
den = [1, 18, 218, 2036, 9085, 25250]

tout, yout = sig.step((num, den), T=t)

plt.figure(figsize = (10, 7))
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('y(t)')
plt.title('Part 2 Task 3')