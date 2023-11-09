import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

w = 150e-6
a = 10e-6
def P(x):
    return np.exp(-2*(x/w)**2)

def slit(x0):
    return scipy.integrate.quad(P, x0-a, x0+a)[0] / scipy.integrate.quad(P, -5*w, 5*w)[0]

slit = np.vectorize(slit)


X = np.linspace(-4*10e-6,4*10e-6,9)
plt.plot(X, slit(X))
plt.show()