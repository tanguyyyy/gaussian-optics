import enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import pandas as pd

#Global parameters
FREQ = 50e9
LAMBDA = 3e8/FREQ
OMEGA_01 = 8e-3
L_x = 0.10
L_z = 0.50

#Number of pixels relative to the y and z axis
N_x = 100
N_z = 1000


#Automatically computed parameters
k = 2*np.pi/LAMBDA
beams = []
optical_systems = []



x_ = np.linspace(-L_x,L_x, N_x)
z_ = np.linspace(0,L_z, N_z)
X, Z = np.meshgrid(x_, z_)
FIELD = np.zeros((N_z,N_x), dtype='complex128')



  




class Optical_system():
    def __init__(self, A, B, C, D, pos, diameter=0.1) -> None:
        self.A, self.B, self.C, self.D = A, B, C, D
        self.pos = pos
        optical_systems.append(self)
        self.diameter = diameter


class Gaussian_beam:
    def __init__(self, w0, z0=0, range_left = 0) -> None:
        self.w0 = w0
        self.z0 = z0
        self.zr = (np.pi*w0**2)/LAMBDA
        beams.append(self)
        self.range_left = range_left
        OS_candidates = [OS for OS in optical_systems if self.range_left < OS.pos < L_z]
        if OS_candidates:
            self.final_optical_system = OS_candidates[np.argmin([OS.pos for OS in OS_candidates])]
            self.range_right = self.final_optical_system.pos
        else:
            self.final_optical_system = None
            self.range_right = L_z

    def omega(self, z):
        return self.w0 * np.sqrt(1 + ((z-self.z0)/self.zr)**2)

    def R(self, z):
        return z - self.z0 + (self.zr**2)/(z-self.z0)
    
    def phi_0(self, z):
        return np.arctan((z-self.z0)/self.zr)

    def E(self, r, z):
        return (2/(np.pi*self.omega(z)**2))**0.5 * np.exp(-r**2/(self.omega(z)**2) - 1j*k*(z-self.z0)*0 - (1j*np.pi*r**2)/(LAMBDA*self.R(z)) + 1j*self.phi_0(z)/2)

    def plot(self, intensity=False, L_x = 10*LAMBDA, L_z = 20*LAMBDA):
        x_ = np.linspace(-L_x,L_x, 10)
        z_ = np.linspace(-L_z,L_z, 20)
        X, Z = np.meshgrid(x_, z_)
        if intensity: U = np.abs(self.E(X,Z))
        else:U = np.real(self.E(X,Z)) 
        plt.pcolormesh(Z,X,U,shading="gouraud")
        plt.axis('scaled')
        plt.show()

def plot_intensity():
    global FIELD
    for Beam in beams:
        z_range = (Z[:,0] <= Beam.range_right) * (Z[:,0] > Beam.range_left)
        filter = np.diag(z_range)
        U = filter @ Beam.E(X,Z)
        FIELD += U
    plt.pcolormesh(Z*1e2,X*1e2,np.abs(FIELD),shading="gouraud")
    plt.axis('scaled')
    plt.show()

def plot_field():
    global FIELD
    for Beam in beams:
        I = np.eye()
        U = Beam.E(X,Z)
        FIELD += U
    plt.pcolormesh(Z,X,np.real(FIELD),shading="gouraud")
    plt.axis('scaled')
    plt.show()

def compute_beam_transformation():
    for Beam in beams:
        if Beam.final_optical_system:
            Lens = Beam.final_optical_system
            d_in = Lens.pos - Beam.z0
            d_out = -((Lens.A*d_in + Lens.B) * (Lens.C*d_in + Lens.D) + Lens.A*Lens.C*Beam.zr**2) /((Lens.C*d_in + Lens.D)**2  + Lens.C**2 * Beam.zr**2)
            w_out = Beam.w0 / np.sqrt((Lens.C*d_in + Lens.D)**2 + Lens.C**2 * Beam.zr**2)
            B = Gaussian_beam(w0 = w_out, z0 = Lens.pos + d_out, range_left = Lens.pos)


def plot_env():
    style='r'
    plt.gca().set_aspect("equal")
    plt.axis((0, L_z*1e2, -L_x*1e2, L_x*1e2))
    for i, Beam in enumerate(beams):
        z_plot = z_[int(Beam.range_left*N_z/L_z):int(Beam.range_right*N_z/L_z)+1]
        plt.plot(z_plot*1e2, Beam.omega(z_plot)*1e2, style)
        plt.plot(z_plot*1e2, -Beam.omega(z_plot)*1e2, style)
    for OS in optical_systems:
        y_min = 0.5 - 0.25 * OS.diameter / L_x
        y_max = 0.5 + 0.25 * OS.diameter / L_x
        plt.axvline(x=OS.pos*1e2, ymin = y_min, ymax= y_max)
    
    plt.show()

def coupling(Beam1, Beam2):
    x = Beam1.w0 / Beam2.w0
    delta_z = Beam1.z0 - Beam2.z0
    return 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*Beam1.w0*Beam2.w0))**2)

def reset_beams():
    global beams
    beams = []


LENS_THICKNESS = 1.5e-2
LENS_CURVATURE = 0.0556

Optical_system(1, LENS_THICKNESS/1.433,-1/0.1, 1+ (1-1.433)*LENS_THICKNESS / (1.433*LENS_CURVATURE), 0.20, diameter=0.07)
#Optical_system(1,0,-1/0.1,1, 0.30, diameter=0.1)

B1 = Gaussian_beam(w0=OMEGA_01, z0=0)
#B2 = Gaussian_beam(w0=OMEGA_01, z0=0.4, range_left=0.3)

df = pd.read_csv('exp_focale_4.csv', sep=',')


plt.plot(df.Zr, 10**((df.G-52.2)/20), 'k+')



offset = -7.5
x_ = np.linspace(0.4,0.5,1000)+offset*1e-2
K = []

for x in x_:
    B1 = Gaussian_beam(w0=OMEGA_01, z0=0)
    B2 = Gaussian_beam(w0=OMEGA_01, z0=x)
    compute_beam_transformation()
    K.append(coupling(B2, beams[-2]))
    reset_beams()



plt.plot(x_*1e2-offset, K, 'k:')
plt.xlabel('Position récepteur (cm)')
plt.ylabel('Couplage')
plt.title(r"$\omega_0=8 mm$")
plt.show()

print(beams)


"""compute_beam_transformation()
B2 = Gaussian_beam(w0=OMEGA_01, z0=0.46)


plot_env()"""
