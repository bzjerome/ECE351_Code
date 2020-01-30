# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:33:43 2020

@author: jero7605
"""

############################################
#                                          #
# Brady Jerome                             #
# ECE351-52                                #
# Lab 1                                    #
# January 6                                #
#                                          #
############################################


import numpy
import scipy.signal
import time

import numpy as np
import scipy.signal as sig

t = 1
print(t)
print("t =",t)
print('t =' ,t, "seconds")
print("t is now =",t/3 , "\n...and can be rounded using 'round()'",round( t/3 ,4))

print(3**2)

list1 = [0,1,2,3]
print('list1:', list1)
list2 = [[0],[1],[2],[3]]
print('list3:', list2)
list3 = [[0,1],[2,3]]
print('list3:', list3)
array1 = numpy.array([0,1,2,3])
print('array1:', array1)
array2 = numpy.array([[0],[1],[2],[3]])
print('array2:', array2)
array3 = numpy.array([[0,1],[2,3]])
print('array3:', array3)

print(np.pi)    #this prints the values of pi


print(np.arange(4),'\n',
      np.arange(0,2,0.5),'\n',
      np.linspace(0,1.5,4))

list1 = [1,2,3,4,5]
array1 = np.array(list1)
print('list1:', list1[0], list1[4])
print('array1:', array1[0], array1[4])
array2 = np. array([[1,2,3,4,5], [6,7,8,9,10]])
list2 = list(array2)
print('array2:', array2[0,2], array2[1,4])
print('list2:', list2[0], list2[1])

print(array2[:,2], array2[0,:])

print('1x3:', np.zeros(3))
print('2x2:', np.zeros((2,2)))
print('2x3:', np.ones((2,3)))

#==============Plot function==============

import matplotlib.pyplot as plt

steps = 0.1 #step size
x= np.arange(-2,2+steps,steps)
y1 = x + 2
y2 = x**2

#Code for plots
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)  #subplot1: subplot format(row,column,number)
plt.plot(x,y1)  #chooses variables for x and y axis
plt.title('Sample Plots for Lab 1') #sets title

plt.ylabel('Subplot 1')
plt.grid(True)      #shows grid on plot
# 
plt.subplot(3,1,2)      #subplot 2
plt.plot(x,y2)
plt.ylabel('Subplot 2')
plt.grid(which='both')
# 
plt.subplot(3,1,3)      #subplot 3
plt.plot(x,y1,'--r',label='y1')
plt.plot(x,y2,'o',label='y2')  #plots both functions on one plot
plt.axis([-2.5,2.5,-0.5,4.5])   #define axis
plt.grid(True)
plt.legend(loc='lower right')   #where to place the legend
plt.xlabel('x') #labels x axis
plt.ylabel('Subplot 3') #labels y axis
plt.show()  #This shows the plots and MUST BE INCLUDED

cRect = 2 + 3j
print(cRect)

cPol = abs(cRect) * np.exp(1j*np.angle(cRect))
print(cPol)

cRect2 = np.real(cPol) + 1j*np.imag(cPol)
print(cRect2)

print(numpy.sqrt(3*5 - 5*5 + 0j))

import control
from scipy.fftpack import fft, fftshift

# =============================================================================
# range()
# np.arange()
# 
# np.append()
# np.insert()
# np.concatenate()
# np.linspace()
# 
# np.logspace()
# 
# np.reshape()
# np.tanspose()
# len()
# .size
# .shape
# .reshape
# 
# .T
# 
# =============================================================================


#=============== QUESTIONS ================================
#
# Which course are you most excited for in your degree? Which course have you enjoied the most so far?
#       I want to look into quantum mechanics because it seems very interesting
#       My favorite course has been Electromagnetic Theory and Microelectronics because they have interesting concepts
#
#
# Leave any feedback on the clarity of the expectations, instructions, and deliverables
#       I dont have any feedback. Everything so far meets my expectations and the class seems like it will be very enjoyable

