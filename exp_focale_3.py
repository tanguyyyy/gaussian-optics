
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize

FILE = "exp_focale_4.csv"

#Paramètres connus
FREQUENCY = 90e9
LAMBDA = 3e8 / FREQUENCY
INDICE = 1.433

LENS_RADIUS = 0.035

#Paramètres estimés (à optimiser)
CONE_RADIUS = 1.55e-2
R_h_CONE = 4.6e-2
LENS_THICKNESS = 1.5e-2
LENS_CURVATURE = 0.0556
RECEIVER_POSITION = 0.39
SOURCE_POSITION = 0.8
FOCALE = 0.10

#Paramètres déduits
OMEGA_CONE = 6e-3
Zc = np.pi*OMEGA_CONE**2 / LAMBDA

print(f"{LAMBDA=}")
print(f"{CONE_RADIUS=}")
print(f"{OMEGA_CONE=}")
print(f"{Zc=}")



def ABCD(f, lens_thickness=0.015, lens_curvature=0.05):
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
    w_in, offset, gain_bias = list(U)

    f = FOCALE
    lens_curvature = LENS_CURVATURE
    lens_thickness = LENS_THICKNESS
    Ze = SOURCE_POSITION
    Zr = RECEIVER_POSITION

    df = pd.read_csv(FILE, sep=';')
    lens_positions = np.array(df.Zl)*1e-2 - offset
    d_ins = lens_positions - Ze

    A, B, C, D = ABCD(f, lens_thickness, lens_curvature)
    d_outs = d_out(A, B, C, D, lens_positions, w_in)
    w_outs = w_out(A, B, C, D, lens_positions, w_in)

    delta_zs = Zr - (lens_positions + d_outs)
    couplings = K(delta_zs, w_in, w_outs)

    rayleight_range = np.pi*w_in**2 / LAMBDA
    #couplings *= (1-np.exp(-2*LENS_RADIUS**2 * rayleight_range**2/(d_ins*w_in)**2))

    

    measured_gain = np.array(df.GdB) - gain_bias
    measured_coupling = 10**(measured_gain/20)

    #plt.plot(lens_positions, couplings)
    #plt.plot(lens_positions, measured_coupling)
    #plt.show()

    return np.linalg.norm(measured_coupling - couplings)**2






U0 = np.array([5e-3, 0, 60])


res = scipy.optimize.minimize(fun, U0, method='CG')
print(res)
"""
____________________________________________________________________________________________________________________
Modélisation
"""

lens_positions = np.linspace(0.1,0.3,1000)
d_ins = lens_positions - SOURCE_POSITION

INDICE = 1.433
A, B, C, D = ABCD(0.11, LENS_THICKNESS, LENS_CURVATURE)

OMEGA_CONE = 6e-3
Zc = np.pi*OMEGA_CONE**2 / LAMBDA

d_outs = d_out(A, B, C, D, lens_positions, OMEGA_CONE)
w_outs = w_out(A, B, C, D, lens_positions, OMEGA_CONE)

delta_zs = lens_positions + d_outs - RECEIVER_POSITION


couplings = K(delta_zs, OMEGA_CONE, w_outs)

couplings_bis = couplings * (1-np.exp(-2*LENS_RADIUS**2 * Zc**2/(d_ins*OMEGA_CONE)**2))

#plt.plot(lens_positions*1e2, (1-np.exp(-2*LENS_RADIUS**2 * Zc**2/(d_ins*OMEGA_CONE)**2)))
#plt.plot(lens_positions*1e2, couplings, 'k:', label='Gaussian Beam Model without considering lens radius')
#plt.plot(lens_positions*1e2, couplings_bis, 'k--', label='Gaussian Beam Model considering lens radius')
"""
____________________________________________________________________________________________________________________
Mesures
"""

df = pd.read_csv(FILE, sep=';')

lens_positions = np.array(df.Zl)*1e-2 - 0.79

gain = np.array(df.GdB) - 132
coupling = 10**(gain/20)

plt.plot(lens_positions*1e2, coupling, '+k', label="Measured coupling")

"""
____________________________________________________________________________________________________________________

"""

plt.xlabel('Lens position (cm)')
#plt.title(r'Best fit: $\omega_{in} = 5.9$ mm ; $f=15.6$ cm')
#plt.axvline(x=32.5, color="lightgray", label = "waist = lens radius")
plt.legend()
plt.show()

