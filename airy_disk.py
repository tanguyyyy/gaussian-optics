from scipy.special import j1
import numpy as np
import matplotlib.pyplot as plt

I0 = 1
D = 0.010 #aperture diameter
LAMBDA = 449 * 10**(-6)
dtheta = 0.001


airy_angle = 1.22 * LAMBDA / D

def I_b(beta):
    return 4*I0 * (j1(beta)/beta)**2

def Beta(t):
    return np.pi * D * np.sin(t) / LAMBDA

def I_t(t):
    return I_b(Beta(t))


p_inside=[0]
X = np.arange(0,1.5,dtheta)

for x in X[1:]:
    p_inside.append(p_inside[-1]+2*np.pi*I_t(x)*dtheta*x)

p_tot = p_inside[-1]
p_inside = p_inside/p_tot

X = X[:int(len(X)/12)]
p_inside = p_inside[:int(len(p_inside)/12)]


plt.plot(X, p_inside,"k--", label="Power included")
plt.plot(X,I_t(X),'k', label = "Intensity")
plt.axvline(x = airy_angle, color = "r", label = r"$\alpha = 1,22 \times \frac{\lambda}{D}$")
plt.grid()
plt.xlabel(r'$\theta$ (rad)')
plt.legend()
plt.show()

print(p_inside[int(airy_angle/dtheta)])
