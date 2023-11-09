import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit

LAMBDA = 1.6667
k = 2*np.pi/LAMBDA

df = pd.read_csv('banc_1.1\standing_waves_z0_290_step_1', sep='\t', index_col=0)
#df = pd.read_csv('banc_1.1\standing_waves_z0_250', sep='\t', index_col=0)
print(df.columns)

z_ = np.array(df['pos (m)'])*1e3+10
K = 10**(np.array(df['Mesure ch1 (V)'])-2) * 1e-5

def func(x, a, b, A, B, theta):
    #return a*x + b + (A * x + B) * np.cos(2*k*x+theta)
    return a*x + b + A*np.exp(-B*x) * np.cos(2*k*x+theta)

def epsilon_1(x, A, B):
    return (A * x + B) * (A*x+B>0)
    #return A*np.exp(-B*x) * np.cos(2*k*x+theta)

def epsilon_2(x, A, B):
    return A*np.exp(-B*x)

a = lmfit.Parameter('a', value = -0.1)
b = lmfit.Parameter('b', value = 1)
A = lmfit.Parameter('A', value=0)
B = lmfit.Parameter('B', value=4)
theta = lmfit.Parameter('theta', value = -np.pi/2)



params = lmfit.Parameters()
params.add(a)
params.add(b)
params.add(A)
params.add(B)
params.add(theta)
model = lmfit.Model(func)


result = model.fit(K, x=z_, params=params)

print(result.fit_report())


Z = np.linspace(0,50,1000)
#plt.plot(Z, func(Z, **result.params.valuesdict()), 'k--', label='model')
#plt.plot(z_, K, 'ok', label='data points')
plt.plot(Z, epsilon_1(Z, -0.0298,1.07), 'k', linestyle='dotted', label=r'$\epsilon_1(z)$')
plt.plot(Z, epsilon_2(Z, 1.09,0.036), 'k--', label=r"$\epsilon_2(z)$")
plt.xlabel('z (mm)')
plt.ylabel('Power coupling (au)')
plt.legend()
plt.show()