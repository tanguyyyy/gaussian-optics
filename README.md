# gaussian-optics
2 month internship at the LERMA team of the Observatoire de Paris



# 3D field measurements vizualisation

The provided script can help you vizualise the 3D measurements of both phase and amplitude measurements of ABmm. The resulting csv file should look like this:
```csv
	x (m)	y (m)	Mesure ch1 (V)	Mesure ch2 (V)
0	-5.2500e-02	-5.2500e-02	 4.6334e+00	 2.4428e+00
1	-5.0750e-02	-5.2500e-02	 4.7498e+00	 3.4530e+00
2	-4.9000e-02	-5.2500e-02	 4.6741e+00	 4.3458e+00
3	-4.7250e-02	-5.2500e-02	 4.5226e+00	 5.9272e+00
4	-4.5500e-02	-5.2500e-02	 4.2688e+00	 5.9266e+00
5	-4.3750e-02	-5.2500e-02	 4.3560e+00	 5.1008e+00
```

Here are some examples of vizualisation functions you can find in the ThreeD_measurements_viz module:

```python
data = Rectangular_sweep_data("banc_1.2/xy_x0_7905_y0_24716_z0_300_step_05_60_x_60")
data.plot()
 ```
[image](resources/simple_plot.png)