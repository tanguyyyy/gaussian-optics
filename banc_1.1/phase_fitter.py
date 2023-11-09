import pandas as pd
from diffractio import np, plt, sp
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import degrees, mm, nm, um
from scipy.signal import convolve2d
from scipy.optimize import minimize
import lmfit
from math import exp

OMEGA_0 = 1.7*mm
FREQ = 180e9
LAMBDA = 3e8/FREQ * 1e6
Zc = np.pi*OMEGA_0**2 / LAMBDA


df = pd.read_csv('banc_1.1\lentille_mes_xy_z0_270_large_avec_abs', index_col=0, sep='\t')

x_ = np.sort(np.array(list({*df['x (m)']}))) * 1e3
y_ = np.sort(np.array(list({*df['y (m)']}))) * 1e3

step_x = x_[1] - x_[0]
step_y = y_[1] - y_[0]
Nx, Ny = len(x_), len(y_)
X, Y = np.meshgrid(x_, y_)

K_volt = np.array(df['Mesure ch1 (V)']).reshape(Nx, Ny)
K = 10**(K_volt-2) * 1e-5
Phi = np.array(df['Mesure ch2 (V)']).reshape(Nx, Ny) * 36 - 180

Phi *= np.pi/180

def display():
    fig = plt.figure(figsize=(20,7.2))
    plt.subplot(1,2,1)
    plt.pcolormesh(X*1e-3, Y*1e-3, K, cmap='gist_rainbow')
    plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Power coupling')
    plt.colorbar(label='Coupling (arbitrary unit)')
    plt.subplot(1,2,2)
    plt.pcolormesh(X*1e-3, Y*1e-3, Phi, cmap='hsv')
    plt.colorbar(label='Phase (°)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.title('Phase')
    plt.show()

def phi(x, x0, Rx, phi_off):
    return np.cos(-np.pi*(x-x0)**2/(LAMBDA*Rx)+phi_off)

def transverse_analysis(axis='x'):
    if axis == 'x':
        data = np.cos(Phi[:,len(x_)//2])
        u_ = x_
    elif axis == 'y':
        u_ = y_
        data = np.cos(K[len(y_)//2,:])
    x0_param = lmfit.Parameter('x0', value = 0)
    Rx_param = lmfit.Parameter('Rx', value = 40)
    phi_off_param = lmfit.Parameter('phi_off', value=0)
    params = lmfit.Parameters()
    params.add(x0_param)
    params.add(Rx_param)
    params.add(phi_off_param)
    model = lmfit.Model(phi)
    #result = model.fit(data, x=u_, params=params)
    #print(result.fit_report())
    plt.plot(x_, data)
    plt.plot(x_, phi(x_, **params.valuesdict()))
    plt.show()
    
    
transverse_analysis()
#display()