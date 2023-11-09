import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


#Global parameters
FREQ = 150e9
LAMBDA = 3e8/FREQ



"""
________________________________________________________________

"""
df_tr = pd.read_csv('exp_focale_5_transverse_corr.csv', sep=';')



K = 10**(np.array(df_tr.Gdb)/10)

print(df_tr)

def f(x, x0, delta_off, K0):
    return K0*np.exp(-2*((x-x0) /delta_off)**2)

xdata = df_tr['x(mm)']
ydata = K





err = np.array(df_tr.err)
err = 20 * (10*err/np.array(df_tr.Gdb)) / np.log(10)

popt, pcov = curve_fit(f, xdata, ydata, sigma=err)

X = np.linspace(-6,10,100)
print(popt)
print(pcov)

plt.errorbar(xdata, ydata, yerr=err,  fmt='+k', linewidth=0.5, capsize=2)
plt.plot(X, f(X, *popt), 'k--')
plt.show()

"""
________________________________________________________________



#en cm
Zc = 0.12
k = 2*np.pi / LAMBDA

df_ph = pd.read_csv("exp_focale_5_phase.csv", sep=';')

print(LAMBDA)
z0 = 0
print(df_ph)
serie1 = np.array(df_ph.z)[:10]
phi1 = np.array([-2*np.pi*i for i in range(len(serie1))])
#plt.plot(serie1, phi1, '+k')
plt.plot(serie1, -k*(serie1-z0)*1e-3, '+r')
plt.show()



________________________________________________________________



df_z = pd.read_csv("exp_focale_5.csv", sep=';')

print(df_z)

xdata = np.array(df_z.z)
ydata = 10**(np.array(df_z.GdB)/20)

ydata_corr = [(ydata[i+1]+ydata[i])/2 for i,u in enumerate(xdata[:-1])]


def K(z, )

plt.plot(xdata[:-1], ydata_corr, 'k+')


plt.show()
"""