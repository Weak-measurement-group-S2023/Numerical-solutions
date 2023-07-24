import numpy as np
from scipy.constants import c
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sb

def pointeur(sigma, t, tau, wavelength):
    w = 2*np.pi*c/wavelength
    N = np.sqrt(1/(np.sqrt(2*np.pi)*sigma))
    g_t = N*np.exp(-np.square((t+tau)/(2*sigma)))
    return g_t*np.exp(-1j*w*(t+tau))

preselected_angle = np.pi/3
phi_x = 0
phi_y = 0
postselected_angle = np.pi/4

N = 1000000
largeur = 10e-9 #ns
v = 0.6*0.01 #m/s

wavelength_0 = 640e-9
w_0 = 2*np.pi*c/wavelength_0

d_min = -3*5
d_max = 3*5
distances = np.linspace(d_min, d_max, N)
time = np.linspace(d_min/c, d_max/c, N)

l1 = 0.1
l2 = 0.03
path_diff = l1 - l2
tau = path_diff/c

E_ref = pointeur(largeur, time, 0, wavelength_0)

E_1 = pointeur(largeur, time, 0, wavelength_0)
E_2 = pointeur(largeur, time, tau, wavelength_0)
E_weak = (1/4)*(E_1+E_2)
E_ref = E_ref*(1/np.sqrt(2))

I_ref = np.conjugate(E_ref)*E_ref*(1/2)
I_weak = np.conjugate(E_weak)*E_weak*(1/2)

plt.plot(time, I_ref)
plt.plot(time, I_weak)
plt.show()

ref_idx = np.argmax(I_ref)
weak_idx = np.argmax(I_weak)

delay_tau = -time[weak_idx] + time[ref_idx]
print("you did", tau)
print("Temporal delay found to be:", 2*delay_tau)