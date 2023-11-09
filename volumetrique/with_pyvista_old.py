import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

df = pd.read_csv('volumetrique\Volume_test_lentille_desaxee_180GHz_45_45_21_xo_223_yo_207_zo_250_steps_05_05_5', sep='\t')

print(len(df))
class Volumetric_data:
    def __init__(self, path, sep='\t') -> None:
        self.df = pd.read_csv(path, sep=sep, index_col=0)
        print(self.df.columns)
        self.x_ = np.sort(np.array(list({*self.df['      x (m)']})))
        self.y_ = np.sort(np.array(list({*self.df['      y (m)']})))
        self.z_ = np.sort(np.array(list({*self.df['      z (m)']})))
        self.step_x = self.x_[1] - self.x_[0]
        self.step_y = self.y_[1] - self.y_[0]
        self.step_z = self.z_[1] - self.z_[0]
        self.Nx, self.Ny, self.Nz = len(self.x_), len(self.y_), len(self.z_)
        self.X, self.Y, self.Z = np.meshgrid(self.x_, self.y_, self.z_)
        self.ndarray_ch1 = np.array(self.df[' Mesure (V)']).reshape(self.Nx, self.Ny, self.Nz)
    def plot_mri_mode(self, channel=1, on_axis='z'):
        pv.set_plot_theme('dark')
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        plotter = pv.Plotter()
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0,0,0)  # The bottom left corner of the data set
        grid.spacing = (self.step_x*1e3, self.step_y*1e3, self.step_z*1e3)  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        plotter.add_mesh_clip_plane(grid, assign_to_axis=on_axis,scalar_bar_args={'title': 'Gain (ua)'})
        plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        plotter.show()

    def plot_slices_mode(self, n_slices = None):
        if not n_slices:
            n_slices = self.Nz
        values = self.ndarray_ch1
        pv.set_plot_theme('dark')
        plotter = pv.Plotter()
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0,0,0)  # The bottom left corner of the data set
        grid.spacing = (self.step_x*1e3, self.step_y*1e3, self.step_z*1e3)  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        slices = grid.slice_along_axis(n=n_slices, axis="z")
        slices.plot(show_edges=False, show_grid=True, scalar_bar_args={'title': 'ABmm output (V)'}, opacity=1)

    def plot_plane(self, n=0, channel=1):
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        X, Y = np.meshgrid(self.x_, self.y_)
        print(values[:,:,n].shape, values.shape)
        plt.pcolormesh(X*1e3, Y*1e3, values[:,:,n], cmap='hsv')
        plt.axis('equal')
        plt.xlabel('x(mm)')
        plt.ylabel('y(mm)')
        plt.title(f"Plane z={self.z_[n]*1e3} mm")
        if channel == 1:
            plt.colorbar(label='Amplitude (ua)')
        else:
            plt.colorbar(label='Phase (°)')
        plt.show()
    

data = Volumetric_data('volumetrique\Volume_test_lentille_desaxee_180GHz_45_45_21_xo_223_yo_207_zo_250_steps_05_05_5')

data.plot_slices_mode()
