import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mesure_waist_180.csv', sep=';')

X = np.array(df['x(mm)'])

K = 10**(np.array(df.GdB)/20)

plt.plot(X, K, '+')
plt.show()