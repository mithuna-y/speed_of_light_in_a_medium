import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# graph parameters
L = 200  # Length of the box
T = 100  # Total time of the animation
dt = 0.1  # Time resolution
medium_start = L / 2  # Start of the medium

# Create x and t arrays
x = np.linspace(0, L, 1000)
t = np.arange(0, T, dt)

# Material constants
material_constant = 50  # The larger this number, the change in speed
num_slices = 200  # Number of slices in the medium
resonant_frequency = 4
resonant_angular_frequency = 2 * np.pi * resonant_frequency
c = 20  # Speed of light

# Main frequency of incoming wave and the predicted phase velocity and group velocity
f0 = 2
main_angular_frequency = 2 * np.pi * f0
c_reduced = c / (1 + material_constant * ((resonant_angular_frequency ** 2 + main_angular_frequency ** 2) / (
            resonant_angular_frequency ** 2 - main_angular_frequency ** 2) ** 2))  # Reduced speed of light for the green dot
c_phase = c / (1 + material_constant / (resonant_angular_frequency ** 2 - main_angular_frequency ** 2))
green_starts = (L / 2) / c  # Time at which the green dot (representing the phase velocity) starts moving

# If you'd like to just see what happens to a single plane wave with frequency , keep the following:
frequencies = [f0]
weights = [1]

# If instead you'd like to calculate it for a gaussian then comment out the above and use this instead:
#sigma = 0.32  # Width of the Gaussian.
#frequencies = np.linspace(0, 2*f0, 500)  # We take 500 frequencies linearly spaced between 0 and 2*f0.
#weights = 1/100 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)  # Weights for the cosine waves Gaussian distribution.

angular_frequencies = [2 * np.pi * f for f in frequencies]  # Angular frequencies

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Add grey patch to represent the start of the material
ax.add_patch(Rectangle((medium_start, -2), L, 4, facecolor="lightgray", alpha=1))

# Create line objects for the vacuum and medium waves
vacuum_sum_line, = ax.plot([], [], color="purple")
medium_sum_lines = [ax.plot([], [], color="purple")[0] for _ in range(num_slices)]

# Add dot objects that move at speed c and c_reduced
red_dot, = ax.plot([], [], 'ro')
green_dot, = ax.plot([], [], 'go')
blue_dot, = ax.plot([], [], 'bo')


# Define the wave functions
def wave(x, t, w, c):
    return np.cos(w * (t - x / c))


# Define the function for the medium
def medium_wave(x, t, w, c, slice_number):
    slice_width = (L / 2) / num_slices
    phase = material_constant * (slice_number + 1) * w / (
            resonant_angular_frequency ** 2 - w ** 2 + 0.012) * slice_width / c
    return np.cos(w * (t - x / c) - phase)


# Initialization function
def init():
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)
    for line in [vacuum_sum_line] + medium_sum_lines:
        line.set_data([], [])
    red_dot.set_data([], [])
    green_dot.set_data([], [])
    blue_dot.set_data([], [])
    return [vacuum_sum_line] + medium_sum_lines + [red_dot, green_dot]


# Update function
def update(frame):
    # Calculate the waves in vacuum
    vacuum_waves = [wave(x[x < medium_start], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    vacuum_sum = sum(vacuum_waves)
    # vacuum_sum = wave(x[x < medium_start], frame, main_angular_frequency, c)
    vacuum_sum_line.set_data(x[x < medium_start], vacuum_sum)

    # Calculate the waves in the medium for each slice
    slice_width = L / (2 * num_slices)
    for slice_number in range(num_slices):
        x_slice = x[
            (x >= medium_start + slice_number * slice_width) & (x < medium_start + (slice_number + 1) * slice_width)]
        medium_waves = [medium_wave(x_slice, frame, w, c, slice_number) * weight for w, weight in
                        zip(angular_frequencies, weights)]
        medium_sum = sum(medium_waves)
        # medium_sum = medium_wave(x_slice, frame, main_angular_frequency, c, slice_number)
        medium_sum_lines[slice_number].set_data(x_slice, medium_sum)

    # Update the dots' positions
    red_dot.set_data(frame * c, 1)  # c*dt is the distance covered in one timestep by the red dot
    if frame >= green_starts:
        green_dot.set_data((frame - green_starts) * c_reduced + L / 2,
                           1)  # c_reduced*dt is the distance covered in one timestep by the green dot
        blue_dot.set_data((frame - green_starts) * c_phase + L / 2,
                          1)  # c_reduced*dt is the distance covered in one timestep by the blue dot, position slightly lower
    return [vacuum_sum_line] + medium_sum_lines + [red_dot, blue_dot, green_dot] #green dot isn't necessary for single plane wave


# Create the animation
ani = FuncAnimation(fig, update, frames=t, init_func=init, blit=True, interval=50, repeat=True)

# Show the animation
plt.show()
