
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Constants
x_range = 100
time_range = 10
time_points = 100
speed_of_light = 2
freq1 = 1
freq2 = 2

# wavepacket constants
wave_packet_L = x_range
wave_packet_angular_freq1 = 2 * np.pi * freq1
wave_packet_angular_freq2 = 2 * np.pi * freq2
weight = 1/np.sqrt(2)

# Initialize time and space variables
t = np.linspace(0, time_range, time_points)
x = np.linspace(0, x_range, 200)

# Prepare the figure
fig, ax = plt.subplots()
ax.set_xlim(0, x_range)
ax.set_ylim(-2.0, 2.0)

# Initialize the plot lines for vacuum and medium
line_wave1, = ax.plot([], [], color="blue", linestyle='--', label="Wave 1", alpha=0.2)
line_wave2, = ax.plot([], [], color="red", linestyle='--', label="Wave 2", alpha=0.2)
line_wave_sum, = ax.plot([], [], color="purple", label="Wave sum")
ax.legend()

# Initialize lines
def init():
    line_wave1.set_data([], [])
    line_wave2.set_data([], [])
    line_wave_sum.set_data([], [])
    return line_wave1, line_wave2, line_wave_sum,

# Wave functions
def wave1(x, t):
    return np.cos(wave_packet_angular_freq1 * (x - speed_of_light * t))

def wave2(x, t):
    return np.cos(wave_packet_angular_freq2 * (x - speed_of_light * t))

# Update function for animation
def update(frame):
    # Time for this frame
    t = frame * time_range / time_points

    # Update wave 1
    x_vals = np.linspace(0, x_range, 100)
    y_vals_wave1 = wave1(x_vals, t)
    line_wave1.set_data(x_vals, y_vals_wave1)

    # Update wave 2
    y_vals_wave2 = wave2(x_vals, t)
    line_wave2.set_data(x_vals, y_vals_wave2)

    # Update wave sum
    y_vals_wave_sum = weight * (y_vals_wave1 + y_vals_wave2)
    line_wave_sum.set_data(x_vals, y_vals_wave_sum)

    return line_wave1, line_wave2, line_wave_sum,

# Create animation with slower speed (increased interval)
ani = FuncAnimation(fig, update, frames=time_points, init_func=init, blit=True, interval=800)

# Show the plot
plt.show()
