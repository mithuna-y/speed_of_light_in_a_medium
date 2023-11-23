# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from matplotlib.patches import Rectangle

# Constants
x_range = 100
time_range = 10
time_points = 100
speed_of_light = 2
material_constant = 5
resonant_frequency = 1.5
resonant_angular_frequency = 2 * np.pi * resonant_frequency
num_slices = 3

# wavepacket constants
wave_packet_width = 1
wave_packet_k0 = 1
wave_packet_freq = 1
wave_packet_initial_pos = -speed_of_light * 5
wave_packet_L = x_range
wave_packet_angular_freq = 2 * np.pi * wave_packet_freq
fourier_cutoff = 50

# Initialize time and space variables
t = np.linspace(0, time_range, time_points)
x = np.linspace(0, x_range, 200)

# Prepare the figure
fig, ax = plt.subplots()
ax.set_xlim(0, x_range)
ax.set_ylim(-1.0, 1.0)

# Shading for the medium
rectangle = Rectangle((x_range / 2, -1), x_range / 2, 2, facecolor="lightgray")
ax.add_patch(rectangle)

# Initialize the plot lines for vacuum and medium
line_vacuum, = ax.plot([], [], color="blue", label="Vacuum")
line_medium, = ax.plot([], [], color="green", label="Medium")
line_vacuum_continued, = ax.plot([], [], color="blue", linestyle='--', label="Vacuum continued")
ax.legend()

# Initialize lines
def init():
    line_vacuum.set_data([], [])
    line_medium.set_data([], [])
    line_vacuum_continued.set_data([], [])
    return line_vacuum, line_medium, line_vacuum_continued,

# Helper functions to calculate Fourier components
def cos_fourier_component(function, index):
    def fc(x):
        return function(x) * np.cos(index * np.pi * x / wave_packet_L)

    return quad(fc, -wave_packet_L, wave_packet_L)[0] / wave_packet_L


def sin_fourier_component(function, index):
    def fs(x):
        return function(x) * np.sin(index * np.pi * x / wave_packet_L)

    return quad(fs, -wave_packet_L, wave_packet_L)[0] / wave_packet_L


# Initial wavepacket at time t
def initial_wavepacket(x, t=5):
    return ((2 * np.pi * wave_packet_width ** 2) ** (-1 / 4) *
            np.exp(-((x - wave_packet_initial_pos - speed_of_light * t) ** 2) /
                   (4 * wave_packet_width ** 2 * t ** 2 + 0.01) +
                   1j * (wave_packet_k0 * (x - speed_of_light * t) - wave_packet_angular_freq * t)))


# Vacuum wavepacket function
def wavepacket_vacuum(x, t):
    sum_vacuum = quad(initial_wavepacket, -wave_packet_L, wave_packet_L)[0] / wave_packet_L
    for index in range(1, fourier_cutoff + 1):
        a_n = cos_fourier_component(initial_wavepacket, index)
        b_n = sin_fourier_component(initial_wavepacket, index)
        sum_vacuum += a_n * np.cos(
            index * np.pi * x / wave_packet_L - (speed_of_light * index * np.pi / wave_packet_L) * t) + b_n * np.sin(
            index * np.pi * x / wave_packet_L - (speed_of_light * index * np.pi / wave_packet_L) * t)
    return sum_vacuum


# Medium wavepacket function
def wavepacket_medium(x, t):
    sum_medium = 0
    for index in range(1, fourier_cutoff + 1):
        a_n = cos_fourier_component(initial_wavepacket, index)
        b_n = sin_fourier_component(initial_wavepacket, index)
        phase = material_constant * wave_packet_angular_freq / (
                    resonant_angular_frequency ** 2 - wave_packet_angular_freq ** 2 + 0.012) * (x_range / 2) / speed_of_light
        sum_medium += a_n * np.cos(index * np.pi * x / wave_packet_L - (
                    speed_of_light * index * np.pi / wave_packet_L) * t - phase) + b_n * np.sin(
            index * np.pi * x / wave_packet_L - (speed_of_light * index * np.pi / wave_packet_L) * t)
    return sum_medium


# Update function for animation
def update(frame):
    # Update vacuum part
    x_vacuum = np.linspace(0, x_range / 2, 100)
    y_vacuum = wavepacket_vacuum(x_vacuum, frame)
    line_vacuum.set_data(x_vacuum, y_vacuum)

    # Update medium part
    x_medium = np.linspace(x_range / 2, x_range, 100)
    y_medium = wavepacket_medium(x_medium, frame)
    line_medium.set_data(x_medium, y_medium)

    # Update vacuum continued part
    y_vacuum_continued = wavepacket_vacuum(x_medium, frame)
    line_vacuum_continued.set_data(x_medium, y_vacuum_continued)

    return line_vacuum, line_medium, line_vacuum_continued,

# Create animation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

# Show the plot
plt.show()