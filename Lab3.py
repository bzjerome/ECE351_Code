# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:40:46 2020
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


plt.rcParams.update({'font.size': 14}) #Set font size in plots

steps = 1e-2    # Define step size
t = np.arange(0, 5+steps, steps)

print('Number of elements: len(t) = ', len(t), '\nFirst Element: t[0] = ',t[0],
      '\nLast Element: t[len(t) - 1] = ', t[len(t) -1 ])
# Notice the array might be a different size than expected since Python starts
# at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array . Notice the array goes from 0 to len () - 1

# --- User - Defined Function ---

# Create output y(t) using a for loop and if/ else statements
def example1(t):    # The only variable sent to the function is t
    y = np.zeros(t.shape)   # initialize y(t) as an array of zeros
    
    for i in range(len(t)): # run the loop once for each index of t
        if i < (len(t) + 1)/3:
            y[i] = t[i]**2
        else:
            y[i] = np.sin(5*t[i]) + 2
    return y # send back the output stored in an array

y = example1(t) # call example1 function with good resolution

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')

t = np.arange(0, 5 + 0.25, 0.25)    # redefine t with poor resolution
y = example1(t) # calls up example1 with poor resolution

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Poor Resolution')
plt.xlabel('t')
plt.show()

#------------------------------------ PART 1 --------------------------------

#------- Creating cosine function ---------
steps = 1e-3    # Define step size
t = np.arange(0, 10+steps, steps)

def func1(t):
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if i < (len(t) + 1)/3:
            x[i] = t[i]**2
        else:
            x[i] = np.cos(5*t[i]) + 2
    return x

x = func1(t) # call func1 function with good Resolution

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')

t = np.arange(0, 10 + 0.25, 0.25)    # redefine t with poor resolution
x = func1(t) # calls up func1 with poor resolution

plt.subplot(2, 1, 2)
plt.plot(t, x)
plt.grid()
plt.ylabel('x(t) with Poor Resolution')
plt.xlabel('t')
plt.show()

#----------------------------------- PART 2 --------------------------------

#--------- Step function ---------
steps = 1e-3
t = np.arange(-1, 1, steps)

def step(t):
    u = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] < 0:
            u[i] = 0
        else:
            u[i] = 1
    return u

u = step(t) # calls the step function

plt.figure(figsize = (5,7))
plt.subplot(2, 1, 1)
plt.plot(t, u)
plt.grid()
plt.ylabel('u(t)')
plt.title('Step function')
plt.xlabel('t')
plt.show()

#---------- Ramp function ---------
steps = 1e-3
t = np.arange(-1, 1, steps)

def ramp(t):
    r = np.zeros((len(t),1))
    
    for i in range(len(t)):
        if t[i] < 0:
            r[i] = 0
        else:
            r[i] = t[i]
    return r

r = ramp(t) # calls the ramp function

plt.figure(figsize = (5,7))
plt.subplot(2, 1, 2)
plt.plot(t, r)
plt.grid()
plt.ylabel('r(t)')
plt.title('Ramp function')
plt.xlabel('t')
plt.show()

#---------- Plotting the given function ---------
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(2, 1, 1)
plt.plot(t,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Plotted Function')
plt.xlabel('t')
plt.show()

#---------- Plot derivative ----------
steps = 1e-3
t = np.arange(-5, 10, steps)

y = plot(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)

plt.figure(figsize = (5,7))
plt.subplot(2, 1, 2)
plt.plot(t, y, '--', label = 'y(t)')
plt.plot(t[range(len(dy))],  dy[:,0], label = 'dy/dt') # dy[:,0] might not work. Take out if needed
plt.grid()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Plotted Derivative Function')
plt.legend()
plt.ylim([2,10])
plt.show()

#--------------------------------- PART 3 ------------------------------------

#--------- TASK 1 -------------#
#                              #
#       Time reversal          #
#                              #
#------------------------------#
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(5, 2, 1)
plt.plot(-t,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Time Reversal')
plt.xlabel('t')
plt.show()

#--------- TASK 2 -------------#
#                              #
#        Time shift            #
#                              #
#------------------------------#

# f(t-4)
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(5, 2, 2)
plt.plot(t-4,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Time Shift f(t-4)')
plt.xlabel('t')
plt.show()

# f(-t-4)
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(5, 2, 3)
plt.plot(-t-4,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Time Shift f(t-4)')
plt.xlabel('t')
plt.show()

#--------- TASK 3 -------------#
#                              #
#        Time scale            #
#                              #
#------------------------------#

# f(t/2)
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(5, 2, 4)
plt.plot(t/2,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Time Scale f(t/2)')
plt.xlabel('t')
plt.show()

# f(2t)
steps = 1e-3
t = np.arange(-5, 10, steps)

def plot(t):
    return (ramp(t) - ramp(t-3) + 5*step(t-3) - 2*step(t-6) - 2*ramp(t-6))

p = plot(t)

plt.figure(figsize = (5,7))
plt.subplot(5, 2, 5)
plt.plot(2*t,p)
plt.grid()
plt.ylabel('p(t)')
plt.title('Time Scale f(2t)')
plt.xlabel('t')
plt.show()
