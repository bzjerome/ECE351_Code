# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:28:31 2020

@author: bzjer
"""

############################################
#                                          #
# Brady Jerome                             #
# ECE351-52                                #
# Lab 8                                    #
# 4/2/2020                                 #
# Fourier Series Approximation of a Square #
# Wave                                     #
#                                          #
############################################

import numpy as np
import matplotlib.pyplot as plt

#%% Defining the values for a[k] and b[k]

a = np.zeros((4, 1))
b = np.zeros((4, 1))

# Define b[k] for the range of k
for k in np.arange(1, 4):
    b[k] = 2/(k*np.pi)*(1 - np.cos(k*np.pi))
    
#Define a[k] for the range of k
for k in np.arange(1, 4):
    a[k] = 0        # Since this is an odd function, a[k] will always be 0 for any value of k

# Print the first values of a[k] and b[k]
a0 = a[0]
a1 = a[1]
b1 = b[1]
b2 = b[2]
b3 = b[3]
print('a0 = ', a0, '\na1 = ', a1, '\nb1 = ', b1, '\nb2 = ', b2, '\nb3 = ', b3)

#%% Graphing x(t) for N = [1, 3, 15, 50, 150, 1500]

T = 8
steps = 1e-5
t = np.arange(0, 20 + steps, steps)
y = 0

N = [1, 3, 15, 50, 150, 1500]

for h in [1, 2]:
    for i in ([1+(h-1)*3, 2+(h-1)*3, 3+(h-1)*3]):
        for k in np.arange(1, N[i - 1] +1):
            b = 2/(k*np.pi)*(1 - np.cos(k*np.pi))
            x = b*np.sin(k*(2*np.pi/T)*t)
            y = y + x
        plt.figure(h, figsize = (10, 7))
        plt.subplot(3, 1, i - (h - 1)*3)
        plt.plot(t, y)
        plt.grid()
        plt.ylabel('N = %i' % N[i - 1])     # Label the y label with the value of N
        if i == 1 or i == 4:
            plt.title('Fourier Series Approximations of x(t)')
        if i == 3 or i == 6:
            plt.xlabel('t[s]')
            plt.show()
        y = 0