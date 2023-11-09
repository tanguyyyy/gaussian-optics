from lecture_mesures import *

data = Rectangular_sweep_data("banc_1.2/xy_x0_7905_y0_24716_z0_300_step_05_60_x_60")
data.plot()
data = Volumetric_sweep_data('banc_1.2\scan_nuit')

data.plot_mri_mode(channel=2, on_axis='x')
data.plot_mri_mode(channel=2)

data.plot_slices_mode(channel=2, n_slices=5)

data.plot_plane(8, channel=2)

data.plane_plot_3D(n=1, channel=1)