
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize


#Paramètres connus
FREQUENCY = 90e9
LAMBDA = 3e8 / FREQUENCY
INDICE = 1.433

LENS_RADIUS = 0.05

#Paramètres estimés (à optimiser)
CONE_RADIUS = 1.55e-2
R_h_CONE = 4.6e-2
LENS_THICKNESS = 3.2e-2
LENS_CURVATURE = 0.0535
RECEIVER_POSITION = 0.70
SOURCE_POSITION = 0.09
FOCALE = 0.15

#Paramètres déduits
OMEGA_CONE = 6e-3
Zc = np.pi*OMEGA_CONE**2 / LAMBDA

print(f"{LAMBDA=}")
print(f"{CONE_RADIUS=}")
print(f"{OMEGA_CONE=}")
print(f"{Zc=}")



def ABCD(f, lens_thickness=0.02, lens_curvature=0.05):
    #return 1, 0, -1/f, 1
    return 1, lens_thickness/INDICE, -1/f, 1+((1-INDICE)*lens_thickness / (INDICE*lens_curvature))


def d_out(A, B, C, D, d_in, w_in):
    """
    d_in: la distance entre le waist incident et le SO
    w_in: la taille du waist du faisceau incident
    """
    z_c = np.pi*w_in**2 / LAMBDA #La distance de Rayleight pour le faisceau incident
    return -((A*d_in + B) * (C*d_in + D) + A*C*Zc**2) /((C*d_in + D)**2  + C**2 * z_c**2)

def w_out(A, B, C, D, d_in, w_in):
    """
    d_in: la distance entre le waist incident et le SO
    w_in: la taille du waist du faisceau incident
    """
    z_c = np.pi*w_in**2 / LAMBDA #La distance de Rayleight pour le faisceau incident
    return w_in / np.sqrt((C*d_in + D)**2 + C**2 * z_c**2)


def K(delta_z, w1, w2):
    """
    Renvoie le couplage pour deux faisceau de même axe décalés de waist différents et dont pour
    la position des waists est différente.
    """
    x = w1 / w2
    return 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*w1*w2))**2)


def fun(U):
    w_in = U

    f = FOCALE
    lens_curvature = LENS_CURVATURE
    lens_thickness = LENS_THICKNESS
    Ze = SOURCE_POSITION
    Zr = RECEIVER_POSITION

    df = pd.read_csv('exp_focale_2_corr.csv', sep=';')
    lens_positions = np.array(df.Zl)*1e-2
    d_ins = lens_positions - Ze

    A, B, C, D = ABCD(f, lens_thickness, lens_curvature)
    d_outs = d_out(A, B, C, D, lens_positions, w_in)
    w_outs = w_out(A, B, C, D, lens_positions, w_in)

    delta_zs = Zr - (lens_positions + d_outs)
    couplings = K(delta_zs, w_in, w_outs)

    rayleight_range = np.pi*w_in**2 / LAMBDA
    couplings *= (1-np.exp(-2*LENS_RADIUS**2 * rayleight_range**2/(d_ins*w_in)**2))

    

    measured_gain = np.array(df.G) - 60
    measured_coupling = 10**(measured_gain/20)

    #plt.plot(lens_positions, couplings)
    #plt.plot(lens_positions, measured_coupling)
    #plt.show()

    return np.linalg.norm(measured_coupling - couplings)**2






U0 = np.array(4e-3)


res = scipy.optimize.minimize(fun, U0)
print(res)
"""
____________________________________________________________________________________________________________________
Modélisation
"""

lens_positions = np.linspace(0.15,0.60,1000)
d_ins = lens_positions - SOURCE_POSITION

INDICE = 1.433
A, B, C, D = ABCD(0.156, LENS_THICKNESS, LENS_CURVATURE)

OMEGA_CONE = 0.006
d_outs = d_out(A, B, C, D, lens_positions, OMEGA_CONE)
w_outs = w_out(A, B, C, D, lens_positions, OMEGA_CONE)

delta_zs = lens_positions + d_outs - 0.7


couplings = K(delta_zs, OMEGA_CONE, w_outs)

couplings_bis = couplings * (1-np.exp(-2*LENS_RADIUS**2 * Zc**2/(d_ins*OMEGA_CONE)**2))

plt.plot(lens_positions*1e2, delta_zs)
plt.plot(lens_positions*1e2, 1e-1*w_outs/OMEGA_CONE)
plt.plot(lens_positions*1e2, couplings, 'k:', label='Gaussian Beam Model without considering lens radius')
plt.plot(lens_positions*1e2, couplings_bis, 'k--', label='Gaussian Beam Model considering lens radius')
"""
____________________________________________________________________________________________________________________
Mesures
"""

df = pd.read_csv('exp_focale_2.csv', sep=',')
lens_positions = np.array(df.Zl)*1e-2 

gain = np.array(df.G) - 60
coupling = 10**(gain/20)

plt.plot(lens_positions*1e2, coupling, '+k', label="Measured coupling")

"""
____________________________________________________________________________________________________________________

"""

plt.xlabel('Lens position (cm)')
plt.title(r'Best fit: $\omega_{in} = 5.9$ mm ; $f=15.6$ cm')
plt.axvline(x=32.5, color="lightgray", label = "waist = lens radius")
plt.legend()
plt.show()


"""
D_in = np.linspace(0,0.5,1000)
for f in [0.12,0.13,0.14,0.15,0.16,0.17]:
    plt.plot(D_in, K(D_in, f), label=f"{f=}")
plt.legend()
plt.show()

F = np.linspace(0.13,0.14,1000)

X, Y = np.meshgrid(D_in, F)
Z = K(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

plt.show()
"""
