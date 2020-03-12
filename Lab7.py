# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:48:05 2020

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

numg = [1, 9]
deng = [1, -2, -40, -64]
numa = [1, 4]
dena = [1, 4, 3]
numb = [1, 26, 168]

#%% Part 1 Task 2
[Rg, Pg, _] = sig.tf2zpk(numg, deng)
print("G Zeroes: ", Rg, "\nG Poles: ", Pg)
[Ra, Pa, _] = sig.tf2zpk(numa, dena)
print("A Zeroes: ", Ra, "\nA Poles: ", Pa)
B = np.roots(numb)
print("B Roots: ", B)

#%% Part 1 Task 4

num = sig.convolve(numg, numa)
print("Numerator: ", num)
den = sig.convolve(deng, dena)
print("Denominator: ", den)

tout, yout = sig.step((num, den))

plt.figure(figsize = (10, 7))
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('A(s)*G(s)')
plt.title('Step Response of A(s)*G(s)')

#%% Part 2 Task 2

numTot = sig.convolve(numa, numg)
print("Total Numerator: ", numTot)
denTot = sig.convolve(dena, deng + sig.convolve(numb, numg))
print("Total Denominator: ", denTot)

[Rtot, Ptot, _] = sig.tf2zpk(numTot, denTot)
print("H(s) Zeroes: ", Rtot, "\nH(s) Poles: ", Ptot)

#%% Part 2 Task 4
tout, yout = sig.step((numTot, denTot))

plt.figure(figsize = (10, 7))
plt.plot(tout, yout)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('H(s)')
plt.title('Step Response of H(s)')