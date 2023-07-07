import numpy as np
from scipy.constants import c
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sb

def electric_field_monochromatic(E_0, k, z, t, position_delay):
    l1 = 0.07
    tau = 2*(l1-position_delay)/c
    w_0 = k*c
    return (E_0/np.sqrt(2))*(np.exp(1j*(k*z -w_0*t)) + np.exp(1j*(k*(z) - w_0*(t+tau))))

def intensity_mono(E_0, k, position_delay):
    return ((E_0**2)*(np.cos(position_delay*k) + 1))/2
"""
def intensity_pulsed(largeur, t, z, k, position_delay):
    w_0 = k*c
    tau = np.empty(len(position_delay))
    l1 = 0.07
    #E_t = np.array((len(t), len(position_delay)))
    I = np.empty((len(t), len(position_delay)))
    for i in range(len(t)):
        for j in range(len(position_delay)):
            tau[j] = 2*(l1 - position_delay[j])/c
            E_0 = np.sqrt(1/(np.sqrt(2*np.pi)*largeur))*np.exp(-np.square(((t[i]+tau[j])-(z)/c)/(2*largeur)))
            E_t = (E_0/np.sqrt(2))*(np.exp(1j*(k*(z) -w_0*t[i])) + np.exp(1j*(k*z - w_0*(t[i]+tau[j]))))
            I[i][j] = np.conjugate(E_t)*E_t
    return I
"""

def int_pulse_1D(largeur, t, z, k, position_delay):
    w_0 = k*c
    tau = np.empty(len(position_delay))
    l1 = 0.07
    #E_t = np.array((len(t), len(position_delay)))
    I = np.empty(len(position_delay))
    for j in range(len(position_delay)):
        tau[j] = 2*(l1 - position_delay[j])/c
        E_0 = np.sqrt(1/(np.sqrt(2*np.pi)*largeur))*np.exp(-np.square(((t+tau[j])-(z)/c)/(2*largeur)))
        E_t = (E_0/np.sqrt(2))*(np.exp(1j*(k*(z) -w_0*t)) + np.exp(1j*(k*z - w_0*(t+tau[j]))))
        I[j] = np.conjugate(E_t)*E_t
    return I


preselected_angle = np.pi/3
phi_x = 0
phi_y = 0
postselected_angle = np.pi/4

N = 100
largeur = 10e-9 #ns
v = 0.6*0.01 #m/s

wavelength_0 = 640e-9
w_0 = 2*np.pi*c/wavelength_0

d_min = -3
d_max = 3
distances = np.linspace(d_min, d_max, N)
time = np.linspace(d_max/c + d_min/c, d_max/c+d_max/c, N)

#-------------------------------------------MONOCHROMATIQUE 1 SEULE FREQUENCE----------------------------------------------------------------------

HeNe_laser = electric_field_monochromatic(E_0=1, k=2*np.pi/wavelength_0, z=0, t=time, position_delay=(distances))
HeNe_intensity = (np.conjugate(HeNe_laser)*HeNe_laser)/2

HeNe_intensity_max = np.max(HeNe_intensity)

HeNe_spectrum = fft(HeNe_intensity)
#freq_axis = fftfreq(len(time), 1 / N)


#----------------------------------------------------MONOCHROMATIQUE MULTI SEULE FREQUENCE-----------------------------------------------------------


"""
HeNe_laser_w = electric_field_monochromatic(E_0=1, k=(2*np.pi)/wavelength, z=0, w=w, t=time, position_delay=(2*v*w*time/c))
HeNe_intensity_w = (np.conjugate(HeNe_laser_w)*HeNe_laser_w)/2

plt.figure()
plt.plot(time, HeNe_intensity_w)
plt.show()

"""

#------------------------------------------DIODE 1 SEULE FREQUENCE-----------------------------------------------------------



#diode_intensity = intensity_pulsed(largeur=largeur, t=time, z=0, k=2*np.pi/wavelength_0, position_delay=(distances))

l1 = 0.07
diode_intensity = int_pulse_1D(largeur=largeur, t=0, z=0, k=2*np.pi/wavelength_0, position_delay=(distances))

Z = diode_intensity
diode_spectrum = fft(diode_intensity)

#-----------------------------------------DIODE MULTI SEULE FREQUENCE-------------------------------------------------------

"""
diode_laser_w = electric_field_pulsed(largeur=largeur, t=time, z=0, k=2*np.pi/wavelength, w=w, position_delay=(2*v*w*time/c))

diode_intensity_w = (np.conjugate(diode_laser_w)*diode_laser_w)/2

plt.figure()
plt.plot(time, diode_intensity_w)
plt.show()

"""

n = np.arange(N)
sr=2000
T = N/sr
freq_axis = 2*np.pi*n/T 


fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0,0].plot(l1-distances, HeNe_intensity)
axs[0,0].set_xlabel('distance')
axs[0,0].set_ylabel('intensity')

axs[0,1].plot(freq_axis, np.abs(HeNe_spectrum))
axs[0,1].set_xlabel('Frequency')
axs[0,1].set_ylabel('Power Spectrum')

axs[1,0].plot(l1-distances, Z)
axs[1,0].set_xlabel('distance')
axs[1,0].set_ylabel('intensity')

axs[1,1].plot(freq_axis, np.abs(diode_spectrum))
axs[1,1].set_xlabel('Frequency')
axs[1,1].set_ylabel('Power Spectrum')

plt.savefig('intensity_et_spectrum.png')
plt.tight_layout()
plt.show()

#-------------------------------------------VISIBILITY MONOCHROMATIQUE 1 SEULE FREQUENCE----------------------------------------------------------------------
l1 = 0.07
#distances = np.linspace(0, 0.14, N)
tau = 2*(l1-distances)/c
w_0 = 2*np.pi*c/wavelength_0

HeNe_laser_0 = electric_field_monochromatic(E_0=1, k=2*np.pi/wavelength_0, z=0, t=0, position_delay=0)
HeNe_I1 = np.conjugate(HeNe_laser_0)*HeNe_laser_0
HeNe_I2 = (np.conjugate(HeNe_laser)*HeNe_laser)

def g_1_mono(w_0, tau):
    return np.cos(w_0*tau)

HeNe_g_1 = g_1_mono(w_0, tau)
HeNe_g_1 = np.sqrt(np.conjugate(HeNe_g_1)*HeNe_g_1)

HeNe_vis = ((2*np.sqrt(HeNe_I1)*np.sqrt(HeNe_I2))/(HeNe_I1+HeNe_I2))*(HeNe_g_1)


#------------------------------------------VISIBILITY DIODE 1 SEULE FREQUENCE-----------------------------------------------------------
tau_0 = np.linspace(0, 0, N)
diode_laser_0 = int_pulse_1D(largeur, 0, 0, 2*np.pi/wavelength_0, tau_0)
diode_I1 = np.conjugate(diode_laser_0)*diode_laser_0
diode_I2 = diode_intensity
def g_1_pulse(w_0, tau, largeur):
    return np.exp(-np.square(tau/(np.sqrt(8)*largeur)))*g_1_mono(w_0, tau)

diode_g_1 = g_1_pulse(w_0, tau, largeur)
diode_g_1 = np.sqrt(np.conjugate(diode_g_1)*diode_g_1)
diode_vis = ((2*np.sqrt(diode_I1)*np.sqrt(diode_I2))/(diode_I1+diode_I2))*(diode_g_1)



fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0,0].plot(tau, np.sqrt(np.conjugate(HeNe_g_1)*HeNe_g_1))
axs[0,0].set_xlabel('tau')
axs[0,0].set_ylabel('g_1')

axs[0,1].plot(distances, HeNe_vis)
axs[0,1].set_xlabel('distance')
axs[0,1].set_ylabel('visibility')

axs[1,0].plot(tau, np.sqrt(np.conjugate(diode_g_1)*(diode_g_1)))
axs[1,0].set_xlabel('tau')
axs[1,0].set_ylabel('g_1')

axs[1,1].plot(distances, diode_vis)
axs[1,1].set_xlabel('distance')
axs[1,1].set_ylabel('visibility')

plt.tight_layout()
plt.savefig('g_1_et_visibility.png')
plt.show()
