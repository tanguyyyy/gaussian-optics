import pandas as pd
from diffractio import np, plt, sp
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio import degrees, mm, nm, um
from scipy.signal import convolve2d
from scipy.optimize import minimize
import lmfit
from math import exp
from scipy.fft import fft2
from scipy.fft import ifft2
from scipy.fft import fftfreq
from scipy.fft import fftshift
import read_measurements

OMEGA_0 = 1*1e-3
FREQ = 180e9
LAMBDA = 3e8/FREQ * 1e6
Zc = np.pi*OMEGA_0**2 / LAMBDA


df = pd.read_csv("Analyse_mesures/exemples_mesures/Thick_Lens_f_50mm_Transmitter_Big_Corrugated_Receiver_Small_Pyramidal_x0_83mm46_y0_237mm45_z0_300mm_dx_0mm2_dy_0mm2_nbx_201_nby_201_tintegr_0s3.txt", index_col=0, sep='\t')


def E0(x,y):
    #Le champ u en z=0
    return (2/(np.pi*OMEGA_0**2))**0.5 * np.exp(-(x**2+y**2)/(OMEGA_0**2))


def display():
    fig = plt.figure(figsize=(20,7.2))
    plt.subplot(1,2,1)
    plt.pcolormesh(X*1e-3, Y*1e-3, K, cmap='gist_rainbow')
    plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Power coupling')
    plt.colorbar(label='Coupling (arbitrary unit)')
    plt.subplot(1,2,2)
    plt.pcolormesh(X*1e-3, Y*1e-3, Phi, cmap='hsv')
    plt.colorbar(label='Phase (°)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.title('Phase')
    plt.show()





def E(p, m, w0, z, A=1, delta=0):
    u = Scalar_source_XY(x=x_, y=y_, wavelength=LAMBDA)
    u.laguerre_beam(A=A,
                    n=p,
                    l=m,
                    r0=(0,0),
                    w0=w0,
                    z0=0,
                    z=z)
    k = 2*np.pi/LAMBDA
    return u.u*np.exp(1j*k*z)


def transverse_analysis(axis='x'):
    if axis == 'x':
        Ku = K[:,len(x_)//2]
        u_ = x_
    elif axis == 'y':
        u_ = y_
        Ku = K[len(y_)//2,:]
    def gauss(x, delta_off, P0, x0, offset):
        return P0*(np.exp(-2*((x-x0)/delta_off)**2) + offset)
    delta_param = lmfit.Parameter('delta_off', value = 4*mm, min=0)
    P0_param = lmfit.Parameter('P0', value = 2000, min=0)
    x0_param = lmfit.Parameter('x0', value = 0, max=2*mm)
    offset_param = lmfit.Parameter('offset', value = 0)
    params = lmfit.Parameters()
    params.add(delta_param)
    params.add(P0_param)
    params.add(x0_param)
    params.add(offset_param)
    model = lmfit.Model(gauss)
    result = model.fit(Ku, x=u_, params=params)
    print(result.fit_report())
    plt.plot(u_*1e-3, Ku, '+k')
    plt.plot(u_*1e-3, gauss(u_, **result.params.valuesdict()), '-k')
    plt.show()
    return result.params.valuesdict()

#On trouve le omega qui convient:


#K *= fft2(E0(X,Y))

def trim_zero_empty(x):
    """
    
    Takes a structure that represents an n dimensional example. 
    For a 2 dimensional example it will be a list of lists.
    For a 3 dimensional one it will be a list of list of lists.
    etc.
    
    Actually these are multidimensional numpy arrays but I was thinking
    in terms of lists.
    
    Returns the same structure without trailing zeros in the inner
    lists and leaves out inner lists with all zeros.
    
    """
    
    if len(x) > 0:
        if type(x[0]) != np.ndarray:
            # x is 1d array
            return list(np.trim_zeros(x))
        else:
            # x is a multidimentional array
            new_x = []
            for l in x:
               tl = trim_zero_empty(l)
               if len(tl) > 0:
                   new_x.append(tl)
            return new_x
    else:
        # x is empty list
        return x
       
def deconv(a, b):
    """
    
    Returns function c such that b * c = a.
    
    https://en.wikipedia.org/wiki/Deconvolution
    
    """
    
    # Convert larger polynomial using fft

    ffta = fftshift(np.fft.fftn(a))
    #ffta *= circular_mask(61,15)
    # Get it's shape so fftn will expand
    # smaller polynomial to fit.
    
    ashape = np.shape(a)
    
    # Convert smaller polynomial with fft
    # using the shape of the larger one

    fftb = fftshift(np.fft.fftn(b,ashape))
    
    # Divide the two in frequency domain

    fftquotient = ffta / fftb
    
    # Convert back to polynomial coefficients using ifft
    # Should give c but with some small extra components

    c = fftshift(np.fft.ifftn(fftquotient))
    
    # Get rid of imaginary part and round up to 6 decimals
    # to get rid of small real components

    #trimmedc = np.around(np.real(c),decimals=6)
    
    # Trim zeros and eliminate
    # empty rows of coefficients
    
    #cleanc = trim_zero_empty(trimmedc)            
    return np.array(c)

def circular_mask(N, n):
    S = np.zeros((N,N))
    n0 = N//2
    for i in range(N):
        for j in range(N):
            S[i,j] = (i-n0)**2+(j-n0)**2 > n**2
    return S

data = read_measurements.Rectangular_sweep_data(df)

K = 10**(data.ndarray_ch1-2)
Phi = (data.ndarray_ch2 * 36 - 180) * np.pi / 180


cos_phi = np.cos(Phi) 
fft = fft2(cos_phi) * circular_mask(201,130)

champ = np.sqrt(K)*np.exp(1j*Phi)

X, Y = data.X, data.Y


test = deconv(K, E0(X,Y))


retest = convolve2d(E0(X,Y), test, mode='same')

Xl, Yl = np.meshgrid(np.linspace(-1,1,121),np.linspace(-1,1,121))



ax = plt.axes(projection='3d')
ax.plot_surface(X*1e3, Y*1e3, np.arccos(ifft2(fft)), cmap='viridis')
#ax.plot_surface(X, Y, E_guess, cmap='viridis')
plt.show()
