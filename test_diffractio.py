from diffractio import um, nm, mm, np
from diffractio.scalar_fields_X import Scalar_field_X
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_fields_XZ import Scalar_field_XZ
import matplotlib.pyplot as plt

x = np.linspace(-50 * mm, 50 * mm, 512)
z = np.linspace(0* um, 400 * mm, 2048)
wavelength = 1.67 * mm

u0 = Scalar_source_X(x, wavelength)
u0.gauss_beam(x0=0, w0=27.7*mm, z0=0)



s0 = Scalar_mask_X(x, wavelength)
s0.lens(x0=100, radius=80 * mm, focal= 100*mm)
u1 = u0 * s0
u1.RS(z=100*mm)

u2 = Scalar_field_XZ(x, z, wavelength=wavelength)
u2.incident_field(u1)


print(u2.u.shape)


u2.draw(logarithm=True)
#s0.draw(kind='phase')
#plt.plot(u2.x*1e-3, np.abs(u2.u), label='rs')
#plt.legend()
plt.show()
