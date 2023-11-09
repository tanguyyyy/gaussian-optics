#==============================================================================
#Importations
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import scipy.integrate

#==============================================================================
#Paramètres globaux
FREQ = 180e9
LAMBDA = 3e8/FREQ
OMEGA_0 = 1.7e-3
#OMEGA_0 Correspond au waist radius des cornets d'émission et de réception.
Zc = np.pi*OMEGA_0**2 / LAMBDA
k = 2*np.pi/LAMBDA #ATTENTION à ce nom de variable global à ne pas utiliser
#                               pour autre chose.

#==============================================================================
#Définition du meshing
L_x = 50e-3 #Distance en mètres couverte selon l'axe x
L_z = 300e-3 #Distance en mètres couverte selon l'axe z (axe de propagation)
N_x = 100 #Nombre de pixels selon l'axe x
N_z = 1000 #Nombre de pixels selon l'axe z

x_ = np.linspace(-L_x,L_x, N_x)
z_ = np.linspace(0,L_z, N_z)
X, Z = np.meshgrid(x_, z_)
FIELD = np.zeros((N_z,N_x), dtype='complex128') #C'est dans cette matrice que
#                                                 sont stockées les valeurs
#                                                 du champ complexe E.


#==============================================================================
#Ces deux listes vont permettre de stocker les instances des classes 
# Optical_system et Gaussian_beam.
BEAMS = []
OPTICAL_SYSTEMS = []


#==============================================================================
#Définition des classes

class Optical_system():
    """
    Cette classe renferme les caractéristiques d'un système optique dont on 
    connaît les paramètres ABCD.
    Attributs:
        self.A: paramètre A (float64)
        self.B: paramètre B (float64)
        self.C: paramètre C (float64)
        self.D: paramètre D (float64)
        self.pos: position sur l'axe optique z en m (float64)
        self.diameter: diamètre en m
    """
    def __init__(self, A, B, C, D, pos, diameter=0.1) -> None:
        self.A, self.B, self.C, self.D = A, B, C, D
        self.pos = pos
        OPTICAL_SYSTEMS.append(self)
        self.diameter = diameter

class Parabolic_mirror():
    """
    À retravailler un jour
    """
    def __init__(self, pos, PFL, EFL, diameter) -> None:
        self.A, self.B, self.C, self.D = 1, PFL-EFL, -1/PFL, 1+(EFL-PFL)/PFL
        self.pos = pos
        self.PFL = PFL
        self.EFL = EFL
        self.diameter = diameter
        OPTICAL_SYSTEMS.append(self)

class Thin_lens():
    """
    Permet de définir une lentille plus simplement à partir de sa position, de
    sa focale, de son diamètre.
    Note: le diamètre ne sert que à visualiser la taille de la lentille. Il 
            n'est pas (encore) pris en compte dans les calculs.
    """
    def __init__(self, pos, f, diameter) -> None:
        self.pos = pos
        self.A, self.B, self.C, self.D = 1, 0, -1/f, 1
        self.diameter = diameter
        OPTICAL_SYSTEMS.append(self)

class Gaussian_beam:
    """
    Un faisceau gaussien en 1D.
    Attributs:
        self.w0: le waist radius
        self.z0: la position du waist sur l'axe optique
        self.x0: la position du waist sur l'axe x perpendiculaire à l'axe optique. 
                    /!\ Toujours mettre x0=0 pour un faisceau émis
        self.zr: la distance de Rayleigh
        self.theta: l'angle d'ouverture
        self.range_left: position sur l'axe z à partir de laquelle le faisceau "existe"
        self.final_optical_system: l'instance de Optical_system qui sera rencontrée en 
                                    premier par le faisceau.
                                    Vaut None si le faisceau ne rencontre rien.
        self.range_right: la fin du domaine d'existence sur l'axe z du faisceau en m
    """
    def __init__(self, w0, z0=0, x0=0, range_left = 0) -> None:
        self.w0 = w0
        self.z0 = z0
        self.x0 = x0 #The offset of the beam relative to the x axis
        self.zr = (np.pi*w0**2)/LAMBDA
        self.theta = np.arctan(LAMBDA / (np.pi * self.w0))
        BEAMS.append(self)
        self.range_left = range_left
        OS_candidates = [OS for OS in OPTICAL_SYSTEMS if self.range_left < OS.pos < L_z]
        if OS_candidates:
            self.final_optical_system = OS_candidates[np.argmin([OS.pos for OS in OS_candidates])]
            self.range_right = self.final_optical_system.pos
        else:
            self.final_optical_system = None
            self.range_right = L_z

    def omega(self, z):
        """
        Taille du waist en fonction de la position z sur l'axe optique
        """
        return self.w0 * np.sqrt(1 + ((z-self.z0)/self.zr)**2)

    def R(self, z):
        """
        Rayon de courbure
        """
        return z - self.z0 + (self.zr**2)/(z-self.z0+1e-15) #a small term is added to avoid a division by zero error
    
    def phi_0(self, z):
        """
        Phase de Gouy
        """
        return np.arctan((z-self.z0)/self.zr)

    def E(self, x, z):
        """
        Champ complexe à la position (x,z)
        """
        return (2/(np.pi*self.omega(z)**2))**0.25 * np.exp(-(x-self.x0)**2/(self.omega(z)**2) - 1j*k*(z-self.z0) - (1j*np.pi*(x-self.x0)**2)/(LAMBDA*self.R(z)) + 1j*self.phi_0(z)/2)

#==============================================================================
#Fonctions de visualisation


def plot_norm():
    """
    Affiche la norme du champ
    """
    global FIELD
    for Beam in BEAMS:
        z_range = (Z[:,0] <= Beam.range_right) * (Z[:,0] > Beam.range_left)
        filter = np.diag(z_range)
        U = filter @ Beam.E(X,Z)
        FIELD += U
    plt.pcolormesh(Z*1e2,X*1e2,np.abs(FIELD),shading="gouraud")
    plt.xlabel("z (cm)")
    plt.ylabel("x (cm)")
    plt.axis('scaled')

def plot_field():
    """
    Représentation de la partie réelle du champ E.
    """
    global FIELD
    for Beam in BEAMS:
        z_range = (Z[:,0] <= Beam.range_right) * (Z[:,0] > Beam.range_left)
        filter = np.diag(z_range)
        U = filter @ Beam.E(X,Z)
        FIELD += U
    plt.pcolormesh(Z*1e2,X*1e2,np.real(FIELD),shading="gouraud")
    plt.axis('scaled')
    plt.xlabel("z (cm)")
    plt.ylabel("x (cm)")
    plt.colorbar()

def plot_env():
    """
    Représenation de l'enveloppe en 1/e du champ E.
    """
    style='r'
    plt.gca().set_aspect("equal")
    plt.axis((0, L_z*1e2, -L_x*1e2, L_x*1e2))
    for i, Beam in enumerate(BEAMS):
        z_plot = z_[int(Beam.range_left*N_z/L_z):int(Beam.range_right*N_z/L_z)+1]
        plt.plot(z_plot*1e2, (Beam.omega(z_plot) + Beam.x0)*1e2, style)
        plt.plot(z_plot*1e2, (-Beam.omega(z_plot) + Beam.x0)*1e2, style)
    for OS in OPTICAL_SYSTEMS:
        y_min = 0.5 - 0.25 * OS.diameter / L_x
        y_max = 0.5 + 0.25 * OS.diameter / L_x
        plt.axvline(x=OS.pos*1e2, ymin = y_min, ymax= y_max)
    plt.xlabel("z (cm)")
    plt.ylabel("x (cm)")


#==============================================================================
#Fonctions de calculs et de simulation

def compute_beam_transformation():
    """
    Calcul la propagation des faisceaux de la gauche vers la droite et crée
    des nouvelles instances de Gaussian_beam.
    """
    for Beam in BEAMS:
        if Beam.final_optical_system:
            Lens = Beam.final_optical_system
            d_in = Lens.pos - Beam.z0
            d_out = -((Lens.A*d_in + Lens.B) * (Lens.C*d_in + Lens.D) + Lens.A*Lens.C*Beam.zr**2) /((Lens.C*d_in + Lens.D)**2  + Lens.C**2 * Beam.zr**2)
            w_out = Beam.w0 / np.sqrt((Lens.C*d_in + Lens.D)**2 + Lens.C**2 * Beam.zr**2)
            B = Gaussian_beam(w0 = w_out, z0 = Lens.pos + d_out, range_left = Lens.pos)



def coupling(Beam1, Beam2):
    """
    Calcul le couplage entre deux instances de faisceaux à partir des
    de Goldsmith (4.16) et (4.30).
    """
    x = Beam1.w0 / Beam2.w0
    delta_z = Beam1.z0 - Beam2.z0
    delta_x0 = Beam1.x0 - Beam2.x0
    K1 = 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*Beam1.w0*Beam2.w0))**2)
    delta_off = np.sqrt(((Beam1.w0**2 + Beam2.w0**2)**2 + 
                        (LAMBDA*delta_z/np.pi)**2)/(Beam1.w0**2 + Beam2.w0**2))
    K2 = np.exp(-2*(delta_x0/delta_off)**2)
    return K1 * K2


def numerical_coupling(Beam1, Beam2, z=0):
    """
    Idem fonction "coupling", sauf qu'on le calcul numériquement au niveau d'un
        plan z choisi. Cela permet de vérifier que les deux fonctions donnent la
        même chose, et permettra de calculer des couplages entre des faisceaux 
        quelconques. 
    On peut aussi vérifier que le plan choisi n'affecte par le résultat.

    Attention: on considère que le faisceau est en 2D et symétrique.
    """
    def fx(x, z):
        return np.real(Beam1.E(x, z) * np.exp(1j*k*(z-Beam1.z0)) * 
                                np.conj(Beam2.E(x, z)*np.exp(1j*k*(z-Beam2.z0))))
    def gx(x, z):
        return np.imag(Beam1.E(x, z) * np.exp(1j*k*(z-Beam1.z0)) * 
                                np.conj(Beam2.E(x, z)*np.exp(1j*k*(z-Beam2.z0))))
    c_x = scipy.integrate.quad(fx, -np.inf, np.inf, args=z)[0]
    c_x += 1j*scipy.integrate.quad(gx, -np.inf, np.inf, args=z)[0]

    #We have to consider the y-axis differently since there is no offset on this axis
    def fy(y, z):
        return np.real(Beam1.E(y+Beam1.x0, z) * np.exp(1j*k*(z-Beam1.z0)) *
                    np.conj(Beam2.E(y+Beam2.x0, z)*np.exp(1j*k*(z-Beam2.z0))))
    def gy(y, z):
        return np.imag(Beam1.E(y+Beam1.x0, z) * np.exp(1j*k*(z-Beam1.z0)) * 
                    np.conj(Beam2.E(y+Beam2.x0, z)*np.exp(1j*k*(z-Beam2.z0))))
    c_y = scipy.integrate.quad(fy, -np.inf, np.inf, args=z)[0]
    c_y += 1j*scipy.integrate.quad(gy, -np.inf, np.inf, args=z)[0]

    return np.abs(c_x)**2 * np.abs(c_y)**2
    


#==============================================================================
#Fonctions de sweep pour avoir le couplage en fonction de la position du
# récepteur

def receptor_sweep_x(lower_bound, upper_bound, z0, N=100):
    """
    Ne marche pas trop...
    Balayage d'un cornet de réception de waist OMEGA_0 selon l'axe x
    Entrées:
        lower_bound: le x où commence le balayage
        upper_bound: le x où fini le balayage
        z0: la position sur l'axe z du balayage
        N: le nombre de points de calcul.
    Sorties:
        X0: les coordonnées de points de calculs sur l'axe x
        output: la valeur du couplage pour ces différents x
    """
    global BEAMS
    X0 = np.linspace(lower_bound, upper_bound, N)
    B_out = BEAMS[-1]
    output = []
    for x0 in X0:
        B_receiver = Gaussian_beam(OMEGA_0, z0, x0)
        output.append(coupling(B_out, B_receiver))
    return (X0, output)

def receptor_sweep_z(lower_bound, upper_bound, x0=0, N=100):
    """
    Ne marche pas trop
    Balayage d'un cornet de réception de waist OMEGA_0 selon l'axe x
    Entrées:
        lower_bound: le z où commence le balayage
        upper_bound: le z où fini le balayage
        x0: la position sur l'axe x du balayage
        N: le nombre de points de calcul.
    Sorties:
        Z0: les coordonnées de points de calculs sur l'axe z
        output: la valeur du couplage pour ces différents z
    """
    global BEAMS
    Z0 = np.linspace(lower_bound, upper_bound, N)
    B_out = BEAMS[-1]
    output = []
    for z0 in Z0:
        B_receiver = Gaussian_beam(OMEGA_0, z0, x0)
        output.append(coupling(B_out, B_receiver))
    return (Z0, output)

def receptor_sweep_xz(xmin, xmax, zmin, zmax, N=50):
    """
    Balayage et affichage 2D d'un cornet de réception de waist OMEGA_0 selon l'axe x
        et l'axe z.
    Entrées:
        xmin, xmax, zmin, zmax: les limites du balayage
        N: le nombre de points sur chaque axe
    Sorties:
        X0: les coordonnées de points de calculs sur l'axe x
        output: la valeur du couplage pour ces différents x
    """
    Bout = BEAMS[-1]
    def f(x0, z0):
        x = Bout.w0 / OMEGA_0
        delta_z = Bout.z0 - z0
        delta_x0 = Bout.x0 - x0
        K1 = 4 / ((x + 1/x)**2 + (LAMBDA * delta_z / (np.pi*Bout.w0*OMEGA_0))**2)
        delta_off = np.sqrt(((Bout.w0**2 + OMEGA_0**2)**2 + (LAMBDA*delta_z/np.pi)**2)/(Bout.w0**2 + OMEGA_0**2))
        K2 = np.exp(-2*(delta_x0/delta_off)**2)
        return K1 * K2
        
    X0, Z0 = np.meshgrid(np.linspace(xmin, xmax, N), np.linspace(zmin, zmax, N))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X0*1e3, (Z0-Bout.z0)*1e3, f(X0, Z0), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Offset on x-axis (mm)")
    ax.set_ylabel("Offset on z-axis (mm)")
    ax.set_zlabel("Coupling with the receiver")
    plt.show()

def reset_all():
    """
    Réinitialise tous les faisceaux et les lentilles
    """
    global BEAMS, OPTICAL_SYSTEMS, FIELD
    BEAMS = []
    OPTICAL_SYSTEMS = []
    FIELD = np.zeros((N_z,N_x), dtype='complex128')











