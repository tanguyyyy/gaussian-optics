import numpy as np
import pandas as pd
from with_pyvista import Volumetric_data
import matplotlib.pyplot as plt

data = Volumetric_data("volumetrique\Volume_test_collimation_180GHz_121_121_3_xo_98_yo_222_zo_150_steps_04_04_150")

data.ndarray_ch1 = data.ndarray_ch1
data.ndarray_ch2 = data.ndarray_ch2
#data.ndarray_ch2 -= np.mean(data.ndarray_ch2, axis=2)

x_, y_ = np.meshgrid(data.x_, data.y_)
x_ *= 1e3
y_ *= 1e3

print(data.df.head())

print(data.ndarray_ch1[50,50,0])


#data.plot_plane(n=2, channel=1)

#data.plot_plane(n=2, channel=2)
#ax = plt.axes(projection='3d')
#X,Y = np.meshgrid(data.x_, data.y_)
#ax.plot_surface(X, Y, data.ndarray_ch1[:,:,2], cmap='viridis')
#plt.show()

fig = plt.figure(figsize=(20,7.4))
plt.subplot(1,2,1)
plt.pcolormesh(x_, y_, data.ndarray_ch1[:,:,2], cmap='Greys')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Coupling (au)')
plt.axis('equal')
plt.subplot(1,2,2)
plt.pcolormesh(x_, y_, data.ndarray_ch2[:,:,2], cmap='hsv')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(label='Phase (°)')
plt.axis('equal')
plt.show()
