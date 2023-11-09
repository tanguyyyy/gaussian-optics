import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import lmfit
from mpl_toolkits.mplot3d import Axes3D

lbd = 1.6667e-3
data = {z_dist: pd.read_csv(f'Mesure_waist_180GHz\champ_z{str(z_dist)}_cornets_180GHZ',sep='\t') for z_dist in [150,160,170,180,190,200,210]}

print(data[150].columns)

def gauss(x, delta_off, P0, x0):
    return P0*np.exp(-2*((x-x0)/delta_off)**2)

def plot2din3d(x,y,z):
    ax.plot(x, y, zs=z, zdir='z')
    d_col_obj = ax.fill_between(x, 0.5, y, step='pre', alpha=0.1) 
    ax.add_collection3d(d_col_obj, zs = z, zdir = 'z')


list_delta_off = []

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for z_dist in data.keys():
    df = data[z_dist]
    window = 20
    lissage = np.array((df[' Mesure (V)'].rolling(window, center=True).sum()/window -0.05).fillna(0)) #Moyenne glissante
    X = np.array(df['    pos (m)'])
    plot2din3d(X*1e3, np.array(df[' Mesure (V)']), z_dist)
    boundx = np.arange(0.1,0.15,1e-3)
    X = np.concatenate((-np.flip(boundx), X, boundx))
    bound = np.zeros((50))
    lissage = np.concatenate((bound, lissage, bound))
    fit, var = curve_fit(gauss, X, lissage, p0 = np.array([0.1, 0.7, 0]))
    list_delta_off.append(fit[0])
    print(fit)
    print(var)
    delta_param = lmfit.Parameter('delta_off', value = 0.1, min=0)
    P0_param = lmfit.Parameter('P0', value = 0.7, min=0)
    x0_param = lmfit.Parameter('x0', value = 0)
    
    params = lmfit.Parameters()
    params.add(delta_param)
    params.add(P0_param)
    params.add(x0_param)
    model = lmfit.Model(gauss)
    result = model.fit(lissage, x=X, params=params)
    print(result.fit_report())
    #plt.plot(X, lissage)
    #plt.plot(X,gauss(X, *fit), label=f"{z_dist=}")
    

ax.set_xlabel('x (mm)')
ax.set_ylabel('Coupling (ABmm V unit)')
ax.set_zlabel('z (mm)')
ax.view_init(elev=-60, azim=90)
plt.legend()
plt.show()

list_delta_off = np.array(list_delta_off)


def delta_fun(z, w0, z0):
    return np.sqrt((4*w0**4 + (lbd*(z-z0)/np.pi)**2)/(2*w0**2))


w0_param = lmfit.Parameter('w0', value = 1e-3, min=0)
z0_param = lmfit.Parameter('z0', value = 1e-3)
params = lmfit.Parameters()
params.add(w0_param)
params.add(z0_param)

model = lmfit.Model(delta_fun)

Z = -np.array([150,160,170,180,190,200,210])*1e-3

result = model.fit(list_delta_off, z=Z, params=params, weights=1)


print(result.params.valuesdict())




#
#fit_w0= minimize(delta_fun, Z, np.array(list_delta_off), p0=np.array([1e-3,0]))



plt.plot(Z*1e3, result.best_fit*1e3, 'k--')
plt.plot(Z*1e3, list_delta_off*1e3, '+k')
plt.xlabel('z (mm)')
plt.ylabel(r'$\delta_{offset}$ (mm)')
plt.show()
