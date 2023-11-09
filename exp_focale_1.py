import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

FREQUENCY = 80e9
LAMBDA = 3e8 / FREQUENCY

CONE_RADIUS = 1.55e-2
R_h = 4.6e-2
OMEGA_0 = (0.644*CONE_RADIUS) / (1 + (np.pi*((0.644*CONE_RADIUS)**2)/(LAMBDA*R_h))**2)
Zc = np.pi*OMEGA_0**2 / LAMBDA
LENS_THICKNESS = 3.2e-2

print(f"{LAMBDA=}")
print(f"{CONE_RADIUS=}")
print(f"{OMEGA_0=}")
print(f"{Zc=}")

def K(delta_z, w_out):
    x = OMEGA_0 / w_out
    return 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*OMEGA_0*w_out))**2)


def d_out(L, d_in=0.05, f=0.18, n=1.439, d=LENS_THICKNESS, R=5.35e-2):
    M_lens_1 = np.array([[1,d/n], [-1/f, 1+(1-n)*d/(n*R)]])
    M_lens_2 = np.array([[1+(n-1)*d/(n*R), d/n], [-1/f, 1]])
    M_tot = M_lens_2 @ np.array([[1, 2*L], [0, 1]]) @ M_lens_1
    A, B, C, D = np.ravel(M_tot)
    return -(A*d_in + B)*(C*d_in + D) + A*C*Zc**2 / ((C*d_in + D)**2 + C**2 * Zc**2)

def w_out(L, d_in=0.05, f=0.18, n=1.439, d=LENS_THICKNESS, R=5.35e-2):
    M_lens_1 = np.array([[1,d/n], [-1/f, 1+(1-n)*d/(n*R)]])
    M_lens_2 = np.array([[1+(n-1)*d/(n*R), d/n], [-1/f, 1]])
    M_tot = M_lens_2 @ np.array([[1, 2*L], [0, 1]]) @ M_lens_1
    A, B, C, D = np.ravel(M_tot)
    return OMEGA_0 / np.sqrt((C*d_in + D)**2 + C**2 * Zc**2)



df = pd.read_csv('exp_focale_1.csv', sep=';')
X = df.ZL2*1e-2 - 0.18

df['K'] = 10**(df.G/10)
df['K'] = df.K / max(df.K)



Mod = K(d_out(X)-0.563+0.074, w_out(X))
Mod = Mod / max(Mod)

plt.plot(X*1e2, Mod, 'k--')
plt.plot(X*1e2,df.K, 'k+')
plt.show()

def error(U):
    z, d_in, f, n, d = list(U)
    Mod = K(d_out(z-0.18,d_in,f,n,d), w_out(z,d_in,f,n,d))
    Mod = Mod / Max
    return np.linalg.norm(Mod - df.K)