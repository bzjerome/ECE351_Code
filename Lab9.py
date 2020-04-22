# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:30:04 2020

@author: bzjer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift

#%% Defining the Fast Fourier Transfrom function

def FFT(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = fft(x)  #perform the fast Fourier transform (fft)
    X_fft_shift = fftshift(X_fft)   # shift zero frequency components
                                    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
                                     # signal, (fs is the sampling frequency and
                                     # needs to be defined perviously in your code)
    X_mag = np.abs(X_fft_shift)/N # compute the magnitrudes of the signal
    X_phi = np.angle(X_fft_shift) # compute the phases of the signal
    return X_mag, X_phi, freq

#%% Defining the clean FFT function
    
def clean_FFT(x, fs):
    N = len(x)  # find the length of the signal
    X_fft = fft(x)  #perform the fast Fourier transform (fft)
    X_fft_shift = fftshift(X_fft)   # shift zero frequency components
                                    # to the center of the spectrum
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
                                     # signal, (fs is the sampling frequency and
                                     # needs to be defined perviously in your code)
    X_mag = np.abs(X_fft_shift)/N # compute the magnitrudes of the signal
    X_phi = np.angle(X_fft_shift) # compute the phases of the signal
    
    for i in range(len(X_phi)):
        if np.abs(X_mag[i]) < 1e-10:    # Allows the phase graph to be readable by setting larger values to 0
            X_phi[i] = 0
            
    return X_mag, X_phi, freq

#%% Part 1 Task 1 (defining and plotting x1)
    
fs = 100
T = 1/fs
t = np.arange(0, 2, T)

x1 = np.cos(2*np.pi*t)
X_mag, X_phi, freq = FFT(x1, fs)    # Runs x1 through the FFT

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x1)
plt.grid()
plt.title('Task1')
plt.ylabel('x1(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X1(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X1(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

#%% Part 1 Task 2 (defining and plotting x2)

x2 = 5*np.sin(2*np.pi*t)
X_mag, X_phi, freq = FFT(x2, fs)    # Runs x2 through the FFT

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x2)
plt.grid()
plt.title('Task2')
plt.ylabel('x2(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X2(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X2(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

#%%Part 1 Task 3 (defining and plotting x3)

x3 = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)**2
X_mag, X_phi, freq = FFT(x3, fs)    # Runs x3 throught the FFT

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x3)
plt.grid()
plt.title('Task3')
plt.ylabel('x3(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X3(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X3(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

#%% Part 1 Task 4 (redefining x1, x2, and x3)
    
# Definging and cleaning x1

fs = 100
T = 1/fs
t = np.arange(0, 2, T)

x1 = np.cos(2*np.pi*t)
X_mag, X_phi, freq = clean_FFT(x1, fs)

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x1)
plt.grid()
plt.title('Task4 x1(t)')
plt.ylabel('x1(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X1(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X1(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

# Defining and plotting clean x2

x2 = 5*np.sin(2*np.pi*t)
X_mag, X_phi, freq = clean_FFT(x2, fs)

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x2)
plt.grid()
plt.title('Task4 x2(t)')
plt.ylabel('x2(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X2(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X2(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

# Defining and plotting clean x3

x3 = 2*np.cos((2*np.pi*2*t)-2) + np.sin((2*np.pi*6*t)+3)**2
X_mag, X_phi, freq = clean_FFT(x3, fs)

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x3)
plt.grid()
plt.title('Task4 x3(t)')
plt.ylabel('x3(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X3(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X3(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()

#%% Part 1 Task 5- Clean Square Wave from Lab 8

T = 8
Ts = 1/fs
t = np.arange(0, 2*T, Ts)
y = 0

N = 15
for k in range(1, N+1):
        b = 2/(k*np.pi)*(1 - np.cos(k*np.pi))
        x = b*np.sin(k*(2*np.pi/T)*t)
        y = y + x
        
x = y
X_mag, X_phi, freq = clean_FFT(x, fs)

plt.figure(figsize = (10,7))
plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.grid()
plt.title('Task5 Clean Square Wave from Lab8')
plt.ylabel('x(t)')
plt.xlabel('t[s]')

plt.subplot(3, 2, 3)
plt.plot(freq, X_mag)
plt.grid()
plt.ylabel('|X(f)|')
plt.subplot(3, 2, 4)
plt.stem(freq, X_mag)
plt.xlim([-2, 2])
plt.grid()

plt.subplot(3, 2, 5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('Phase of X(f)')
plt.xlabel('f[Hz]')
plt.subplot(3, 2, 6)
plt.stem(freq, X_phi)
plt.xlim([-2, 2])
plt.grid()
plt.xlabel('f[Hz]')
plt.show()