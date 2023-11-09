import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import scipy.integrate
from scipy.special import hermite
from math import factorial
import pyvista as pv
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import um, mm, cm, degrees






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

    def E(self,x,y,z):
        return self.Bx.E(x,z) * self.By.E(y,z)


def plot_norm():
    opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
    p = pv.Plotter()
    p.add_volume(np.abs(self.E(X,Y,Z)), cmap="coolwarm", opacity=opacity)
    p.show()


Bx = Gaussian_beam_hermite(0.001,z0=0, x0=0, n=3)
By = Gaussian_beam_hermite(0.001,z0=0, x0=0, n=5)

B = Gaussian_Beam_2D(Bx, By)

from diffractio import degrees, eps, mm, no_date, np, um
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_sources_XY import Scalar_source_XY

x0 = np.linspace(-25 * um, 25 * um, 128)
y0 = np.linspace(-25 * um, 25 * um, 128)
z0 = np.linspace(-100 * um, 100 * um, 256)
wavelength = .6328 * um

t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
t1.circle(
    r0=(0 * um, 0 * um), radius=(10 * um, 10 * um), angle=0 * degrees)
t1.draw(filename='usage10.png')

uxyz = Scalar_mask_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
uxyz.incident_field(u0=t1)

uxyz.RS(verbose=True, num_processors=1)


"""
opacity = [0, 0, 0, 0.1, 0.3, 0.6, 1]
p = pv.Plotter()
p.add_volume(np.abs(uxyz.u), cmap="coolwarm", opacity='geom')
p.show()
"""


print(uxyz.u.shape)


values = np.abs(uxyz.u)
plotter = pv.Plotter()
grid = pv.UniformGrid()
grid.dimensions = np.array(values.shape) + 1
grid.origin = (0,0,0)  # The bottom left corner of the data set
grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
plotter.add_mesh_clip_plane(grid, assign_to_axis='z')
plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
plotter.show()


opacity = [0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
p = pv.Plotter()
p.add_volume(values, cmap="viridis", opacity=opacity)
p.show()