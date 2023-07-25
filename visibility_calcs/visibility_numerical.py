import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
import seaborn as sb


class Pointeur:
    def __init__(self, time, function, delay):
        self.Time = np.array(time)
        self.Function = np.array(function)
        self.Delay = delay

class PolarizationState:
    def __init__(self, horizontal, vertical):
        self.Horizontal = horizontal
        self.Vertical = vertical

def generate_pointer_state(t, pulse_width, wavelength, z):
    amplitude = 1 / np.sqrt(np.sqrt(2*np.pi)*pulse_width)
    w = 2 * np.pi * c / wavelength
    k = (2 * np.pi) / wavelength

    exponential_term = np.exp(-np.power((t - z/c) / (2*pulse_width), 2))
    real_part = amplitude * exponential_term * np.cos((k*z - w*t))
    imag_part = amplitude * exponential_term * np.sin((k*z - w*t))
    pointeur = real_part + 1j * imag_part

    return pointeur

def couple_initial_state(state, pointeur_function, degree_of_freedom):
    coupled_state = np.empty(len(degree_of_freedom), dtype=PolarizationState)
    
    for i in range(len(degree_of_freedom)):
        coupled_state[i] = PolarizationState(
            pointeur_function[i] * state.Horizontal,
            pointeur_function[i] * state.Vertical
        )
    
    return coupled_state

def mach_zedner_interference(coupled_polarisation_state, degree_of_freedom, delay, wavelength, delayed_part):
    partie_H = np.array([state.Horizontal for state in coupled_polarisation_state])
    partie_V = np.array([state.Vertical for state in coupled_polarisation_state])
    
    if delayed_part.lower() in ['h', '0']:
        H_delta = interaction_operator(partie_H, wavelength, delay)  # weakly interacted function
        for i in range(len(coupled_polarisation_state)):
            coupled_polarisation_state[i].Horizontal = H_delta[i]
    
    elif delayed_part.lower() in ['v', '1']:
        V_delta = interaction_operator(partie_V, wavelength, delay)  # weakly interacted function
        for i in range(len(coupled_polarisation_state)):
            coupled_polarisation_state[i].Vertical = V_delta[i]
    
    else:
        raise ValueError("Invalid value for delayed_part. Expected 'H', '0', 'V', '1' for horizontal or vertical polarization.")
    
    return coupled_polarisation_state

def interaction_operator(amplitude, wavelength, delay):
    w = 2*np.pi*c/wavelength
    delayed_amplitude = amplitude * np.exp(-1j*wavelength*delay)
    return delayed_amplitude

def apply_postselection(intermediate_state, postselected_angle, phi_x, phi_y, degree_of_freedom):
    u = np.cos(postselected_angle) * np.cos(phi_x) + 1j * np.cos(postselected_angle) * np.sin(phi_x)
    v = np.sin(postselected_angle) * np.cos(phi_y) + 1j * np.sin(postselected_angle) * np.sin(phi_y)

    polarisation = PolarizationState(u, v)
    
    #print("horizontal polarisation:", polarisation.Horizontal)
    #print("vertical polarisation:", polarisation.Vertical)
    #print("horizontal conj polarisation:", np.conj(polarisation.Horizontal))
    #print("vertical conj polarisation:", np.conj(polarisation.Vertical))
    
    postselected_state = np.empty(len(degree_of_freedom), dtype=PolarizationState)
    for i in range(len(degree_of_freedom)):
        postselected_state[i] = PolarizationState(intermediate_state[i].Horizontal * polarisation.Horizontal,
                                                 intermediate_state[i].Vertical * polarisation.Vertical)
    
    return postselected_state

def degree_of_coherence(postselected_state):
    g_1 = np.empty(len(postselected_state), dtype=complex)
    max_magnitude = 0.0

    # Calculate the complex product and find the maximum magnitude
    for i in range(len(postselected_state)):
        E_1 = postselected_state[i].Horizontal
        E_2 = postselected_state[i].Vertical

        mean_product = E_2 * np.conj(E_1)

        magnitude = np.abs(mean_product)
        if magnitude > max_magnitude:
            max_magnitude = magnitude

        g_1[i] = mean_product

    # Normalize the values in g_1
    g_1 /= max_magnitude

    return g_1

def intensity_HV_Profile(coupled_polarisation_state, degree_of_freedom):
    intensity_profile = np.zeros(len(degree_of_freedom))
    for i in range(len(degree_of_freedom)):
        total_state = (1/np.sqrt(2))*(coupled_polarisation_state[i].Horizontal + coupled_polarisation_state[i].Vertical)
        intensity_profile[i] = np.abs(total_state)**2
        #intensity_H = np.abs(coupled_polarisation_state[i].Horizontal) ** 2
        #intensity_V = np.abs(coupled_polarisation_state[i].Vertical) ** 2
        #intensity_profile[i] = (1 / np.sqrt(2)) * (intensity_H + intensity_V)
    
    return intensity_profile

def main():
    preselected_angle = np.pi/3
    phi_x = 0
    phi_y = 0
    postselected_angle = np.pi/4
    wavelength = 640e-9  #laser pulsé
    largeur = 10e-9 #ns
    N = 100
    distances = np.linspace(0, 1e-2, N)  #de 0 a 10cm
    time = np.linspace(-5*largeur, 5*largeur, N)

    a = np.cos(preselected_angle) * np.cos(phi_x) + 1j * np.cos(preselected_angle) * np.sin(phi_x)
    b = np.sin(preselected_angle) * np.cos(phi_y) + 1j * np.sin(preselected_angle) * np.sin(phi_y)

    pointer = Pointeur(time=time, function=None, delay=None)
    pointer.Function = generate_pointer_state(time, largeur, wavelength=wavelength, z=0)
    polarisation = PolarizationState(a, b)

    I_d = np.empty((len(distances), len(distances)))
    v = np.empty(len(distances))
    for i in range(len(time)):
        for j in range(len(distances)):
            preselected_state = couple_initial_state(polarisation, pointer.Function, time)

            intermediate_state = mach_zedner_interference(preselected_state, time, distances[i]/c, wavelength, 'H')

            postselected_state = apply_postselection(intermediate_state, postselected_angle, 0, 0, time)

            I = intensity_HV_Profile(postselected_state, time)
            I_max = np.max(I)
            I_min = np.min(I)

            vi = (I_max - I_min)/(I_max + I_min)
            v[i] = vi
            I_d[i][j] = I[j]

    X, Y = np.meshgrid(time, distances)
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, I_d, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('time')
    ax.set_ylabel('distance')
    ax.set_zlabel('Intensity')

    # Display the plot
    plt.show()
   
    preselected_state = couple_initial_state(polarisation, pointer.Function, time)

    intermediate_state = mach_zedner_interference(preselected_state, time, 0.003, wavelength, 'H')

    postselected_state = apply_postselection(intermediate_state, postselected_angle, 0, 0, time)

    I = intensity_HV_Profile(postselected_state, time)

    g_1 = degree_of_coherence(postselected_state)

    plt.figure()
    plt.plot(time, g_1)
    plt.xlabel('time')
    plt.ylabel('g_1')
    plt.show()
   
    plt.figure()
    plt.plot(distances, v)
    plt.xlabel('distance')
    plt.ylabel('contrast')
    plt.show()

    I_max = np.max(I)
    I_min = np.min(I)

    v = (I_max - I_min)/(I_max + I_min)
    print('Visibility = ', v)

if __name__ == "__main__":
    main()