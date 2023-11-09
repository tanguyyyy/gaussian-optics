import matplotlib.pyplot as plt
import lmfit
import numpy as np
lbd = 1.6667
OMEGA_0 = 1.7

Z = np.array([300,290,280,270,260,250])

wx = np.array([6.110,7.704,8.517,9.240,10.356,11.302])
err_x = np.array([0.32,0.34,0.49,0.26,0.60,0.56])

wy = np.array([4.679,5.645,6.633,6.797,7.224,8.644])
err_y = np.array([0.31,0.43,0.41,0.56,0.52,0.57])


def delta_fun(z, w0, z0):
    return np.sqrt(((w0**2+OMEGA_0**2)**2 + (lbd*(z0-z)/np.pi)**2)/(w0**2+OMEGA_0**2))
    #return lbd*(z-z0)/np.sqrt(OMEGA_0**2 + w0**2)
    #return w0*(z-z0)


w0_param = lmfit.Parameter('w0', value = 3e-3)
z0_param = lmfit.Parameter('z0', value = 320e-3)
params = lmfit.Parameters()
params.add(w0_param)
params.add(z0_param)

model = lmfit.Model(delta_fun)

result = model.fit(wx, z=Z, params=params, weights=err_x)

print(result.fit_report())


plt.plot(Z, wx, '+k')
plt.plot(Z, wy, '+b')

plt.plot(Z, delta_fun(Z, 3.5, 325),label="3.5")
plt.plot(Z, delta_fun(Z, 3.2, 325), label='3.2')
plt.plot(Z, delta_fun(Z, 4, 325), label='4')
plt.plot(Z, delta_fun(Z, **result.params.valuesdict()),'k--')
plt.legend()
plt.show()




