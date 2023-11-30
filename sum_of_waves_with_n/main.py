import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

#graph parameters
L = 1000 # Length of the box
T = 100  # Total time of the animation
dt = 0.1  # Time resolution
medium_start = 50  # Start of the medium

# Material constants
material_constant = 50
resonant_frequency = 4
resonant_angular_frequency = 2 * np.pi * resonant_frequency

c = 20 # Speed of light

def fourier_weights(f, angular_frequencies, T, num_points=1000):
    """
    Computes the Fourier coefficients for function f at the specified angular frequencies.

    Parameters:
    - f: The input function (time -> value).
    - angular_frequencies: The list of angular frequencies.
    - T: The period over which to compute the Fourier coefficients.
    - num_points: The number of points to use in the numerical approximation.

    Returns:
    - A list of Fourier coefficients corresponding to the input angular frequencies.
    """
    # Create a time array
    t = np.linspace(0, T, num_points)

    # Evaluate the function at these times
    f_values = f(t)

    # Compute the Fourier coefficients using the trapezoid rule
    dt = T / num_points
    weights = []
    for omega in angular_frequencies:
        weight = 2 / T * np.trapz(f_values * np.exp(-1j * omega * t), dx=dt)
        weights.append(weight)

    return weights


# normal distribution
sigma = 0.32
mu = 0
normal_distribution = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x-mu) / sigma) ** 2)
f0 = 1.0
frequencies = np.linspace(0, 2 * f0, 500)
angular_frequencies = [2 * np.pi * f for f in frequencies]
#T = 2 * np.pi / (2 * np.pi * f0)  # One period for the highest frequency
weights = fourier_weights(normal_distribution, angular_frequencies, T)

# or Gaussian wave packet
# Central frequency of the Gaussian.
#f0 = 2
#frequencies = np.linspace(0, 2*f0, 500)  # We take 500 frequencies linearly spaced between 0 and 2*f0.
#
# # Weights for the cosine waves - Gaussian wave packet.
#weights = 1/50 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)
#angular_frequencies = [2 * np.pi * f for f in frequencies]  # Angular frequencies


main_angular_frequency = 2 * np.pi * f0
c_reduced = c/ (1 + material_constant * ((resonant_angular_frequency**2 + main_angular_frequency**2)/(resonant_angular_frequency**2 - main_angular_frequency**2)**2)) # Reduced speed of light for the green dot
c_phase = c/(1 + material_constant / (resonant_angular_frequency**2 - main_angular_frequency**2))
green_starts = medium_start / c  # Time at which the green dot starts moving
alpha = 0.5  # Transparency of individual waves


# Create x and t arrays
x = np.linspace(0, L, 1000)
t = np.arange(0, T, dt)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Add a patch to represent the medium
ax.add_patch(Rectangle((medium_start, -2), L-medium_start, 4, facecolor="lightgray"))


# Create line objects for the vacuum and medium waves
vacuum_sum_line, = ax.plot([], [], color="red")
medium_sum_line, = ax.plot([], [], color="purple")
continued_vacuum_line, = ax.plot([], [], color="red", alpha=alpha)


# Add dot objects that move at speed c and c_reduced
red_dot, = ax.plot([], [], 'ro')
green_dot, = ax.plot([], [], 'go')
blue_dot, = ax.plot([], [], 'bo')

def n(w):
    return 1 + material_constant / (resonant_angular_frequency**2 - w**2 + 0.0012)

# Define the wave functions
def wave(x, t, w, c):
    return np.cos(w * (t - x/c))

# Define the function for the medium
def medium_wave(x, t, w, c):
    return np.where(x <= medium_start,
                    np.cos(w * (t - x/c)),
                    np.cos(w * (t - medium_start / c - n(w) * (x - medium_start)/c)))

# Initialization function
def init():
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)
    vacuum_sum_line.set_data([], [])
    medium_sum_line.set_data([], [])
    continued_vacuum_line.set_data([], [])
    red_dot.set_data([], [])
    green_dot.set_data([], [])
    blue_dot.set_data([], [])
    return [vacuum_sum_line, medium_sum_line, red_dot, green_dot, blue_dot, continued_vacuum_line]


# Update function
def update(frame):
    # Calculate the waves in vacuum
    vacuum_waves = [wave(x[x < medium_start], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    vacuum_sum = sum(vacuum_waves)
    vacuum_sum_line.set_data(x[x < medium_start], vacuum_sum)

    # Calculate the waves in the medium
    medium_waves = [medium_wave(x[x >= medium_start], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    medium_sum = sum(medium_waves)
    medium_sum_line.set_data(x[x >= medium_start], medium_sum)

    # Continued vacuum
    continued_vacuum = [wave(x[x >= medium_start], frame, w, c) * weight for w, weight in
                        zip(angular_frequencies, weights)]
    continued_vacuum_sum = sum(continued_vacuum)
    continued_vacuum_line.set_data(x[x >= medium_start], continued_vacuum_sum)

    # Update the dots' positions
    red_dot.set_data(frame*c, 1)
    if frame >= green_starts:
        green_dot.set_data((frame-green_starts)*c_reduced + medium_start, 1)
        blue_dot.set_data((frame - green_starts) * c_phase + medium_start, 1)
    return [vacuum_sum_line, medium_sum_line, continued_vacuum_line, red_dot, blue_dot]

# Create the animation
ani = FuncAnimation(fig, update, frames=t, init_func=init, blit=True, interval=10, repeat=True)

# Show the animation
plt.show()
