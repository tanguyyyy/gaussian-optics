import numpy as np

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

    ffta = np.fft.fftn(a)
    
    # Get it's shape so fftn will expand
    # smaller polynomial to fit.
    
    ashape = np.shape(a)
    
    # Convert smaller polynomial with fft
    # using the shape of the larger one

    fftb = np.fft.fftn(b,ashape)
    
    # Divide the two in frequency domain

    fftquotient = ffta / fftb
    
    # Convert back to polynomial coefficients using ifft
    # Should give c but with some small extra components

    c = np.fft.ifftn(fftquotient)
    
    # Get rid of imaginary part and round up to 6 decimals
    # to get rid of small real components

    trimmedc = np.around(np.real(c),decimals=6)
    
    # Trim zeros and eliminate
    # empty rows of coefficients
    
    cleanc = trim_zero_empty(trimmedc)
                
    return np.array(cleanc)

