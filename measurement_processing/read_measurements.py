import numpy as np
import pyvista as pv
import pandas as pd
import matplotlib.pyplot as plt


class Rectangular_sweep_data:
    def __init__(self, df) -> None:
        self.df = df
        self.x_ = np.sort(np.array(list({*self.df['x (m)']})))
        self.y_ = np.sort(np.array(list({*self.df['y (m)']})))
        self.step_x = self.x_[1] - self.x_[0]
        self.step_y = self.y_[1] - self.y_[0]
        self.Nx, self.Ny = len(self.x_), len(self.y_)
        self.X, self.Y = np.meshgrid(self.x_, self.y_)
        self.ndarray_ch1 = np.array(self.df['Mesure ch1 (V)']).reshape(self.Nx, self.Ny)
        self.ndarray_ch2 = np.array(self.df['Mesure ch2 (V)']).reshape(self.Nx, self.Ny)

    def plot(self):
        fig = plt.figure(figsize=(20,7.2))
        plt.subplot(1,2,1)
        plt.pcolormesh(self.X*1e3, self.Y*1e3, self.ndarray_ch1, cmap='gist_rainbow')
        plt.axis('equal')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Power coupling')
        plt.colorbar(label='Coupling (ABmm voltage)')
        plt.subplot(1,2,2)
        plt.pcolormesh(self.X*1e3, self.Y*1e3, self.ndarray_ch2, cmap='hsv')
        plt.colorbar(label='Phase (ABmm voltage)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.axis('equal')
        plt.title('Phase')
        plt.show()

class Volumetric_sweep_data:
    def __init__(self, df) -> None:
        self.df = df
        self.x_ = np.sort(np.array(list({*self.df['x (m)']})))
        self.y_ = np.sort(np.array(list({*self.df['y (m)']})))
        self.z_ = np.sort(np.array(list({*self.df['z (m)']})))
        self.step_x = self.x_[1] - self.x_[0]
        self.step_y = self.y_[1] - self.y_[0]
        self.step_z = self.z_[1] - self.z_[0]
        self.Nx, self.Ny, self.Nz = len(self.x_), len(self.y_), len(self.z_)
        self.X, self.Y, self.Z = np.meshgrid(self.x_, self.y_, self.z_)
        self.ndarray_ch1 = np.array(self.df['Mesure ch1 (V)']).reshape(self.Nx, self.Ny, self.Nz)
        self.ndarray_ch2 = np.array(self.df['Mesure ch2 (V)']).reshape(self.Nx, self.Ny, self.Nz)

    def get_rectangular_data(self, n_slice=0):
        chosen_z = self.z_[n_slice]
        new_df = self.df[self.df['z (m)']==chosen_z].copy()
        return Rectangular_sweep_data(new_df)

    def plot_mri_mode(self, channel=1, on_axis='z'):
        """
        Visualisation du type IRM des données
        Entrées:
            channel: 1 ou 2 pour afficher la voie 1 ou 2
            on_axis: 'z' par défaut, on peut mettre aussi 'x' ou 'y'. Correspond à l'orientation
                        de la tranche de coupe.
        """
        pv.set_plot_theme('dark')
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        plotter = pv.Plotter()
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0,0,0)  # The bottom left corner of the data set
        grid.spacing = (self.step_x*1e3, self.step_y*1e3, self.step_z*1e3)  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        if channel == 1:
            plotter.add_mesh_clip_plane(grid, assign_to_axis=on_axis,scalar_bar_args={'title': 'Amplitude (dB)'})
        elif channel == 2:
            plotter.add_mesh_clip_plane(grid, assign_to_axis=on_axis,scalar_bar_args={'title': 'Phase (°)'}, cmap='hsv')
        plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
        plotter.show()

    def plot_slices_mode(self, channel=1, n_slices = None):
        """
        Visualisation du type tranches.
        Entrées:
            channel: 1 ou 2 pour afficher la voie 1 ou 2
            n_slices: un entier correspondant au nombre de tranches à afficher. Si None, affiche toutes les tranches.
        """
        if not n_slices:
            n_slices = self.Nz
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        pv.set_plot_theme('dark')
        plotter = pv.Plotter()
        grid = pv.UniformGrid()
        grid.dimensions = np.array(values.shape) + 1
        grid.origin = (0,0,0)  # The bottom left corner of the data set
        grid.spacing = (self.step_x*1e3, self.step_y*1e3, self.step_z*1e3)  # These are the cell sizes along each axis
        grid.cell_data["values"] = values.flatten(order="F")  # Flatten the array!
        slices = grid.slice_along_axis(n=n_slices, axis="z")
        if channel == 1:
            slices.plot(show_edges=False, show_grid=True, scalar_bar_args={'title': 'Amplitude (dB)'}, opacity=1)
        elif channel == 2:
            slices.plot(show_edges=False, show_grid=True, scalar_bar_args={'title': 'Phase (°)'}, opacity=1, cmap='hsv')


    def plot_plane(self, n=0, channel=1):
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        X, Y = np.meshgrid(self.x_, self.y_)
        fig = plt.figure(figsize=(5,4))
        plt.pcolormesh(X*1e3, Y*1e3, values[:,:,n], cmap='hsv')
        plt.axis('equal')
        plt.xlabel('x(mm)')
        plt.ylabel('y(mm)')
        if channel == 1:
            plt.colorbar(label='Amplitude (ua)')
        else:
            plt.colorbar(label='Phase (°)')
        plt.show()

    def plane_plot_3D(self, n=0, channel=1):
        values = self.ndarray_ch1 * (channel==1) + self.ndarray_ch2 * (channel==2)
        X, Y = np.meshgrid(self.x_, self.y_)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, values[:,:,n], cmap='viridis')
        plt.show()

