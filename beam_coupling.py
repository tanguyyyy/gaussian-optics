import enum
from errno import EFAULT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import scipy.integrate
from scipy.special import hermite
from math import factorial
import pyvista as pv


#Global parameters
FREQ = 180e9
LAMBDA = 3e8/FREQ
OMEGA_0 = 1.7e-3
Zc = np.pi*OMEGA_0**2 / LAMBDA
L_x = 0.005
L_y = 0.005
L_z = 0.010

#Number of pixels relative to the x and z axis
N_x = 100
N_y = 100
N_z = 200

#Automatically computed parameters
#Please keep in mind that those names are used for this global variables!
k = 2*np.pi/LAMBDA
BEAMS = []
OPTICAL_SYSTEMS = []


x_ = np.linspace(-L_x,L_x, N_x)
y_ = np.linspace(-L_y,L_y, N_y)
z_ = np.linspace(-L_z,L_z, N_z)
X, Y, Z = np.meshgrid(x_, y_, z_)

class Gaussian_beam_hermite:
    def __init__(self, w0, n=0, z0=0, x0=0, range_left = 0) -> None:
        self.n = n
        self.w0 = w0
        self.z0 = z0
        self.x0 = x0 #The offset of the beam relative to the 1D transverse axis
        self.zr = (np.pi*w0**2)/LAMBDA

    def omega(self, z):
        return self.w0 * np.sqrt(1 + ((z-self.z0)/self.zr)**2)

    def R(self, z):
        return z - self.z0 + (self.zr**2)/(z-self.z0+1e-15) #a small term is added to avoid a division by zero error
    
    def phi_0(self, z):
        return np.arctan((z-self.z0)/self.zr)

    def E(self, x, z):
        return (2/np.pi)**0.25 * (self.omega(z)*2**self.n * factorial(self.n))**(-0.5) * hermite(self.n)(np.sqrt(2)*x/self.omega(z)) * np.exp(-(x-self.x0)**2/(self.omega(z)**2) - 1j*k*(z-self.z0) - (1j*np.pi*(x-self.x0)**2)/(LAMBDA*self.R(z)) + 1j*(2*self.n+1)*self.phi_0(z)/2)



class Gaussian_Beam_2D:
    def __init__(self, Bx, By) -> None:
        self.Bx = Bx
        self.By = By
        self.x0 = Bx.x0
        self.y0 = By.x0
        BEAMS.append(self)

    def coupling(B1, B2, z=0):
        cx = numerical_coupling_1D(B1.By, B2.By,z)
        cy = numerical_coupling_1D(B1.By, B2.By,z)
        return np.abs(cx)**2 * np.abs(cy)**2

    def E(self,x,y,z):
        return Bx.E(x,z) * By.E(y,z)
        
    def plot_mri_mode(self, on_axis='z'):
        """
        pv.set_plot_theme('dark')
        values = np.abs(self.E(X,Y,Z))
        plotter = pv.Plotter()
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0,0,0)  # The bottom left corner of the data set
        grid.spacing = (N_x/L_x*1e3, N_y/L_y*1e3, N_z/L_z*1e3)  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        plotter.add_mesh_clip_plane(grid, assign_to_axis=on_axis)
        plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        plotter.show()
        """
        opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
        p = pv.Plotter()
        p.add_volume(np.abs(self.E(X,Y,Z)), cmap="coolwarm", opacity=opacity)
        p.show()



def coupling(Beam1, Beam2):
    x = Beam1.w0 / Beam2.w0
    delta_z = Beam1.z0 - Beam2.z0
    delta_x0 = Beam1.x0 - Beam2.x0
    K1 = 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*Beam1.w0*Beam2.w0))**2)
    delta_off = np.sqrt(((Beam1.w0**2 + Beam2.w0**2)**2 + (LAMBDA*delta_z/np.pi)**2)/(Beam1.w0**2 + Beam2.w0**2))
    K2 = np.exp(-2*(delta_x0/delta_off)**2)
    return K1 * K2


def numerical_coupling_1D(Beam1, Beam2, z=0):
    """
    This function is meant to verify the function "coupling" output. It may be useful for more complexe BEAMS.
    z is the reference plane chosen for the computing. It does not affect the result significatively.
    """
    def fx(x, z):
        return np.real(Beam1.E(x, z) * np.exp(1j*k*(z-Beam1.z0)) * np.conj(Beam2.E(x, z)*np.exp(1j*k*(z-Beam2.z0))))
    def gx(x, z):
        return np.imag(Beam1.E(x, z) * np.exp(1j*k*(z-Beam1.z0)) * np.conj(Beam2.E(x, z)*np.exp(1j*k*(z-Beam2.z0))))
    c_x = scipy.integrate.quad(fx, -np.inf, np.inf, args=z)[0] + 1j*scipy.integrate.quad(gx, -np.inf, np.inf, args=z)[0]
    return c_x
    
def numerical_coupling_2D(Beam1, Beam2, z=0):
    """
    Beam1, Beam2: 2D Gaussian beams
    z : float, position on z axis for calculation
    """
    cx = numerical_coupling_1D(Beam1.Bx, Beam2.Bx, z)
    cy = numerical_coupling_1D(Beam1.By, Beam2.By, z)
    return np.abs(cx)**2 * np.abs(cy)**2

def receptor_sweep_x(Bmes, lower_bound, upper_bound, z0, N=100):
    X0 = np.linspace(lower_bound, upper_bound, N)
    output = []
    for x0 in X0:
        Bx = Gaussian_beam_hermite(OMEGA_0, n=0, z0=z0, x0=x0)
        cx = numerical_coupling_1D(Bmes.Bx, Bx)
        output.append(np.abs(cx)**2)
    return (X0, output)

def receptor_sweep_xz(xmin, xmax, zmin, zmax):
    X0, Z0 = np.meshgrid(np.linspace(xmin, xmax, 50), np.linspace(zmin, zmax, 50))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X0*1e3, (Z0-Bout.z0)*1e3, f(X0, Z0), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Offset on x-axis (mm)")
    ax.set_ylabel("Offset on z-axis (mm)")
    ax.set_title("Coupling with the receiver")
    plt.show()

def reset_beams():
    global BEAMS
    BEAMS = []

Bx = Gaussian_beam_hermite(0.001,z0=0, x0=0, n=1)
By = Gaussian_beam_hermite(0.001,z0=0, x0=0, n=5)

B = Gaussian_Beam_2D(Bx, By)


B.plot_mri_mode(on_axis='z')


#X0, K = receptor_sweep_x(B,-0.05,0.05, z0=0.1, N=100)

#plt.plot(X0, K)
#plt.show()

