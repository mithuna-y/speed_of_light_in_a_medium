import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

# Constants and global variables
x_range = 400
c = 20
material_constant = 5
resonant_frequency = 1.5
resonant_angular_frequency = 2 * np.pi * resonant_frequency
num_slices = 30

# Wave packet parameters
sigma = 1
k_0 = 1
frequency = 1
x_0 = -c * 5
L = 100
angular_frequency = 2 * np.pi * frequency
fourier_cutoff = 75

# Create plot
fig, ax = plt.subplots()
ax.set_xlim(0, x_range)
ax.set_ylim(-1.0, 1.0)

# Create shaded area to represent medium
ax.axvspan(x_range/2, x_range, facecolor='gray', alpha=0.5)

# Initialize plot lines
lines = []
for i in range(num_slices + 1):
    color = "blue"
    linestyle = "--" if i < num_slices else "-"
    line, = ax.plot([], [], color=color, linestyle=linestyle)
    lines.append(line)

# Function to initialize the plot
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Define the initial wave packet
def initial_wavepacket(x):
    t = 5  # Set time to a non-zero value to avoid division by zero
    return (2 * np.pi * sigma ** 2) ** (-1 / 4) * np.exp(
            -((x - x_0 - c * t) ** 2) / (4 * sigma ** 2 * t ** 2 + 0.01) + 1j * (
                        k_0 * (x - c * t) - angular_frequency * t))

# Helper functions to calculate Fourier components
def cos_fourier_component(function, index):
    def fc(x):
        return function(x) * np.cos(index * np.pi * x / L)
    return quad(fc, -L, L)[0] / L

def sin_fourier_component(function, index):
    def fs(x):
        return function(x) * np.sin(index * np.pi * x / L)
    return quad(fs, -L, L)[0] / L

# Define the wave packet in vacuum
def wavepacket_vacuum(x, t):
    sum = quad(initial_wavepacket, -L, L)[0] / L
    for index in range(1, fourier_cutoff + 1):
        an = cos_fourier_component(initial_wavepacket, index)
        bn = sin_fourier_component(initial_wavepacket, index)
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t)
    return sum

# Define the wave packet in medium
def wavepacket_medium(x, t, number_of_slices, slice_number):
    n = 50
    sum = 0
    slice_width = (x_range / 2) / number_of_slices
    for index in range(1, n + 1):
        an = cos_fourier_component(initial_wavepacket, index)
        bn = sin_fourier_component(initial_wavepacket, index)
        phase = material_constant * (slice_number+1) * angular_frequency / (
                resonant_angular_frequency ** 2 - angular_frequency ** 2 + 0.012) / c
        sum += an * np.cos(index * np.pi * x / L - (c * index * np.pi / L) * t + phase) + bn * np.sin(
            index * np.pi * x / L - (c * index * np.pi / L) * t + phase)
    return sum


# Create shaded area to represent medium
ax.axvspan(x_range/2, x_range, facecolor='gray', alpha=0.5)

# Initialize plot lines
lines = []
plane_wave_lines = []  # New list for the plane wave lines

# Initialize wave lines
for i in range(num_slices + 1):
    color = "blue"
    linestyle = "--" if i < num_slices else "-"
    line, = ax.plot([], [], color=color, linestyle=linestyle)
    lines.append(line)

# Initialize plane wave lines
for i in range(num_slices + 1):
    color = "red"
    linestyle = "--" if i < num_slices else "-"
    plane_wave_line, = ax.plot([], [], color=color, linestyle=linestyle)
    plane_wave_lines.append(plane_wave_line)

# Function to update the plot for each frame
def update(frame):
    t = frame / 10.0  # Scale time to slow down animation
    slice_width = (x_range / 2) / num_slices

    # Update wave lines
    for slice_number, line in enumerate(lines):
        if slice_number < num_slices:
            x_start = (x_range / 2) + slice_number * slice_width
            x_end = (x_range / 2) + (slice_number + 1) * slice_width
            x_values = np.linspace(x_start, x_end, 100)
            y_values = wavepacket_medium(x_values, t, num_slices, slice_number)
        else:
            x_values = np.linspace(0, x_range / 2, 200)
            y_values = wavepacket_vacuum(x_values, t)
        line.set_data(x_values, y_values)

    # Update plane wave lines
    for slice_number, plane_wave_line in enumerate(plane_wave_lines):
        if slice_number < num_slices:
            x_start = (x_range / 2) + slice_number * slice_width
            x_end = (x_range / 2) + (slice_number + 1) * slice_width
            x_values = np.linspace(x_start, x_end, 100)
            phase = material_constant * (slice_number+1) * resonant_angular_frequency / (
                resonant_angular_frequency ** 2 - angular_frequency ** 2 + 0.012)
            y_values = np.cos(angular_frequency * (t - x_values / c) - phase)
        else:
            x_values = np.linspace(0, x_range / 2, 200)
            y_values = np.cos(angular_frequency * (t - x_values / c))
        plane_wave_line.set_data(x_values, y_values)

    return lines + plane_wave_lines  # Return all lines


# Create animation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)

plt.show()
