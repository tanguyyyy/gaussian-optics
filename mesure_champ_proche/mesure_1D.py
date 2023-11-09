import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve
import lmfit

FREQ = 180e9
LAMBDA = 3e8/FREQ
OMEGA_0 = 1.7e-3
Zr = np.pi * OMEGA_0**2 / LAMBDA

k = 2*np.pi/LAMBDA

df = pd.read_csv('mesure_champ_proche/champ_proche', sep='\t')

print(df.columns)

X = np.array(df['    pos (m)'])





def omega(z):
    return OMEGA_0 * np.sqrt(1 + (z/Zr)**2)

def R(z):
    return z+ (Zr**2)/(z+1e-15) #a small term is added to avoid a division by zero error
    
def phi_0(z):
    return np.arctan(z/Zr)

def E0(x, z):
    #return 1
    return (2/(np.pi*omega(z)**2))**0.25 * np.exp(-x**2/(omega(z)**2) - 1j*k*z - (1j*np.pi*x**2)/(LAMBDA*R(z)) + 1j*phi_0(z)/2)

E0 = np.vectorize(E0)


def fresnel_integral(E0, a, z0):
    def integral(x, z):
        def function(xp):
            r = np.sqrt((x-xp)**2 + z**2)
            return E0(xp, z0) * z * np.exp(1j*k*r) / r**2
        def real(xp):
            return np.real(function(xp))
        def imag(xp):
            return np.imag(function(xp))
        return -(1j / LAMBDA) * (scipy.integrate.quad(real, -a, a)[0] + 1j*scipy.integrate.quad(imag, -a,  a)[0])
    return np.vectorize(integral)



def func(u):
    z, A0, a, z0 = u[0], u[1], u[2], u[3]
    F = fresnel_integral(E0, a, z0)
    #F est le champ à une distance z
    field = A0 * F(X, z)
    conv = np.convolve(np.conj(field), E0(X, z), mode='same')
    conv = np.abs(conv)**2
    mes = np.array(df[' Mesure (V)'])
    return np.linalg.norm(mes-conv)**2

"""
params = lmfit.Parameters()
z_param = lmfit.Parameter('z', value = 40e-3, min=0)
params.add(z_param)
A0_param = lmfit.Parameter('A0', value = 1)
params.add(A0_param)
a_param = lmfit.Parameter('a', value = 20e-3, min=0)
params.add(a_param)
z0_param = lmfit.Parameter('z0', value = 20e-3)
params.add(z0_param)

print(params)

model = lmfit.Model(func)

data = np.array(df[' Mesure (V)'])

result = model.fit(data, x=X, params=params)

print(result.fit_report())
"""


res = minimize(func, x0=np.array([50e-3,1,0.015,30e-3]))
print(res)


