import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2
from scipy.signal import convolve2d
from cmath import sqrt

sqrt = np.vectorize(sqrt)

FREQ = 180e9
LAMBDA = 1.7e8/FREQ
OMEGA_0 = 5e-3
Zc = np.pi*OMEGA_0**2 / LAMBDA
k = 2*np.pi / LAMBDA


L_x = 10e-3
L_y = 10e-3

res_x = LAMBDA/100
res_y = LAMBDA/100

x_ = np.arange(-L_x/2, L_x/2, res_x)
y_ = np.arange(-L_y/2, L_y/2, res_y)

kx_ = 2 * np.pi * np.arange(-0.5/res_x, 0.5/res_x, 1/L_x)
ky_ = 2 * np.pi * np.arange(-0.5/res_y, 0.5/res_y, 1/L_y)

print(len(x_), len(kx_))

X0, Y0 = np.meshgrid(x_, y_)
Kx, Ky = np.meshgrid(kx_, ky_)

def U0(x,y):
    #Le champ u en z=0
    return (2/(np.pi*OMEGA_0**2))**0.5 * np.exp(-(x**2+y**2)/(OMEGA_0**2))

def propagator(kx, ky, z):
    return np.exp(1j*k*sqrt(1-(kx**2+ky**2)/k**2)*z)

def plot_propagator(z):
    ax = plt.axes(projection='3d')
    ax.plot_surface(Kx, Ky, propagator(Kx,Ky,z), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")
    ax.set_title(r"$u(x,y,0)$")
    plt.show()



def plot_U0():
    ax = plt.axes(projection='3d')
    ax.plot_surface(X0*1e3, Y0*1e3, U0(X0,Y0), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("x-axis (mm)")
    ax.set_ylabel("y-axis (mm)")
    ax.set_title(r"$u(x,y,0)$")
    plt.show()

def U(z):
    initial_field = U0(X0, Y0)
    plt.pcolormesh(X0,Y0, np.abs(initial_field))
    plt.show()
    propagator_field = propagator(Kx, Ky, z)
    plt.pcolormesh(X0,Y0, np.abs(propagator_field))
    plt.show()
    ifft = ifft2(propagator_field)
    plt.pcolormesh(X0,Y0, np.abs(ifft))
    plt.show()
    U = convolve2d(initial_field, ifft, mode='same')
    return U

field = U(0.5)
plt.pcolormesh(np.abs(field))
plt.axis('equal')
plt.show()