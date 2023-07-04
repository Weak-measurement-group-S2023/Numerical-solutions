import numpy as np
from scipy.constants import c
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sb

def electric_field_monochromatic(E_0, k, z, w, t, position_delay):
    return (E_0/np.sqrt(2))*(np.exp(1j*(k*z -w*t)) + np.exp(1j*(k*(z+position_delay) - w*t)))

def intensity_mono(E_0, k, position_delay):
    return ((E_0**2)*(np.cos(position_delay*k) + 1))/2

def electric_field_pulsed(largeur, t, z, k, w, position_delay):
    E_0 = np.sqrt(1/(np.sqrt(2*np.pi)*largeur))*np.exp(-np.square((t-(z+position_delay)/c)/(2*largeur)))
    return (E_0/np.sqrt(2))*(np.exp(1j*(k*z -w*t)) + np.exp(1j*(k*(z+position_delay) - w*t)))

preselected_angle = np.pi/3
phi_x = 0
phi_y = 0
postselected_angle = np.pi/4

N = 1000
largeur = 10e-9 #ns
v = 0.6*0.01 #m/s

#wavelength = 640e-9  #laser pulsé
w_0 = 5.65e13
w_f = 6.60e13
w = np.linspace(w_0, w_f, N)

wavelength_0 = (2*np.pi*c)/w_0
wavelength_f = (2*np.pi*c)/w_f
wavelength = np.linspace(wavelength_0, wavelength_f, N)

t_min = -5*largeur
t_max = 5*largeur
time = np.linspace(t_min, t_max, N)

d_min = (t_min*c)*3/1000
d_max = (t_max*c)*3/1000
distances = np.linspace(d_min, d_max, N)
print(d_min)


#-------------------------------------------MONOCHROMATIQUE 1 SEULE FREQUENCE----------------------------------------------------------------------

HeNe_laser = electric_field_monochromatic(E_0=1, k=2*np.pi/wavelength_0, z=0, w=w_0, t=time, position_delay=(2*(0.0762 - (0.0762+distances))*2*np.pi/wavelength_0))
HeNe_intensity = (np.conjugate(HeNe_laser)*HeNe_laser)/2

HeNe_spectrum = fft(HeNe_intensity)


"""
#----------------------------------------------------MONOCHROMATIQUE MULTI SEULE FREQUENCE-----------------------------------------------------------
freq_axis = fftfreq(len(time), 1 / N)



HeNe_laser_w = electric_field_monochromatic(E_0=1, k=(2*np.pi)/wavelength, z=0, w=w, t=time, position_delay=(2*v*w*time/c))
HeNe_intensity_w = (np.conjugate(HeNe_laser_w)*HeNe_laser_w)/2

plt.figure()
plt.plot(time, HeNe_intensity_w)
plt.show()


"""
#------------------------------------------DIODE 1 SEULE FREQUENCE-----------------------------------------------------------

t_min = -5*largeur
t_max = 5*largeur
time = np.linspace(t_min, t_max, N)

d_min = (t_min*v)*3
d_max = (t_max*v)*3
distancess = np.linspace(d_min, d_max, N)
print(d_max)

diode_laser = electric_field_pulsed(largeur=largeur, t=time, z=0, k=2*np.pi/wavelength_0, w=w_0, position_delay=(2*(0.0762 - (0.0762+distancess))*2*np.pi/wavelength_0))

diode_intensity = (np.conjugate(diode_laser)*diode_laser)/2

diode_spectrum = np.abs(fft(diode_intensity))

"""
#-----------------------------------------DIODE MULTI SEULE FREQUENCE-------------------------------------------------------


diode_laser_w = electric_field_pulsed(largeur=largeur, t=time, z=0, k=2*np.pi/wavelength, w=w, position_delay=(2*v*w*time/c))

diode_intensity_w = (np.conjugate(diode_laser_w)*diode_laser_w)/2

plt.figure()
plt.plot(time, diode_intensity_w)
plt.show()
"""



fig, axs = plt.subplots(2, 2, figsize=(12, 8))
freq_axis = fftfreq(len(time), 1 / N)

axs[0,0].plot(distances, HeNe_intensity)
axs[0,0].set_xlabel('distance')
axs[0,0].set_ylabel('intensity')

axs[0,1].plot(freq_axis, np.abs(HeNe_spectrum))
axs[0,1].set_xlabel('Frequency')
axs[0,1].set_ylabel('Power Spectrum')

axs[1,0].plot(distancess, diode_intensity)
axs[1,0].set_xlabel('distance')
axs[1,0].set_ylabel('intensity')

axs[1,1].plot(freq_axis, diode_spectrum)
axs[1,1].set_xlabel('Frequency')
axs[1,1].set_ylabel('Power Spectrum')

plt.tight_layout()
plt.show()