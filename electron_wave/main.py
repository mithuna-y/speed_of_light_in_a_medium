import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Graph parameters
L = 200  # Length of the box
T = 100  # Total time of the animation
dt = 0.1  # Time resolution
medium_start = L / 2  # Start of the medium

# Material constants
material_constant = 50
resonant_frequency = 4
resonant_angular_frequency = 2 * np.pi * resonant_frequency

# Central frequency of the Gaussian.
f0 = 2
main_angular_frequency = 2 * np.pi * f0

# Width of the Gaussian.
sigma = 0.1

c = 20  # Speed of light
c_reduced = c / (1 + material_constant * ((resonant_angular_frequency ** 2 + main_angular_frequency ** 2) / (
            resonant_angular_frequency ** 2 - main_angular_frequency ** 2) ** 2))  # Reduced speed of light for the green dot
green_starts = (L / 2) / c  # Time at which the green dot starts moving
alpha = 0.1  # Transparency of individual waves
num_slices = 100  # Number of slices in the medium

# Frequencies for the cosine waves.
frequencies = np.linspace(0, 2 * f0, 500)  # We take 500 frequencies linearly spaced between 0 and 2*f0.

# Weights for the cosine waves - Gaussian distribution.
weights = 1 / 100 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)
angular_frequencies = [2 * np.pi * f for f in frequencies]  # Angular frequencies

# Create x and t arrays
x = np.linspace(0, L, 1000)
t = np.arange(0, T, dt)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Add patches to represent the medium slices
for slice_number in range(num_slices):
    ax.add_patch(Rectangle((medium_start + slice_number * L / (2 * num_slices), -2), L / (2 * num_slices), 4,
                           facecolor="lightgray", alpha=1 / (slice_number + 1)))

# Create line objects for the vacuum, medium, and electron waves
vacuum_sum_line, = ax.plot([], [], color="red", label="Original Wave")
medium_sum_lines = [ax.plot([], [], color="purple", label="Resulting Wave" if _ == 0 else "")[0] for _ in range(num_slices)]
electron_wave_lines = [ax.plot([], [], color="blue", alpha=alpha, label="Electron Wave" if _ == 0 else "")[0] for _ in range(num_slices)]

# Add dot objects that move at speed c and c_reduced
green_dot, = ax.plot([], [], 'go')


# Define the wave functions
def wave(x, t, w, c):
    return np.exp(1j*w * (t - x / c))


# Define the function for the medium
def medium_wave(x, t, w, c, slice_number):
    slice_width = (L / 2) / num_slices
    phase = material_constant * (slice_number + 1) / (
            resonant_angular_frequency ** 2 - w ** 2 + 0.012) * slice_width / c
    return np.exp(1j*w*((t - x / c) - phase))


# Define the function for the electron wave
def electron_wave(x, t, w, c, slice_number):
    slice_width = (L / 2) / num_slices
    phase = material_constant * (slice_number + 1) / (
            resonant_angular_frequency ** 2 - w ** 2 + 0.012) * slice_width / c
    return - 1j * slice_number * phase * np.exp(1j*w * (t - x / c))

def subtracted_wave(x, t, w, c, slice_number):
    slice_width = (L / 2) / num_slices
    phase = material_constant * (slice_number + 1) / (
            resonant_angular_frequency ** 2 - w ** 2 + 0.012) * slice_width / c
    return medium_wave(x, t, w, c, slice_number) - wave(x, t, w, c)

# Initialization function
def init():
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)
    for line in [vacuum_sum_line] + medium_sum_lines + electron_wave_lines:
        line.set_data([], [])
    green_dot.set_data([], [])
    return [vacuum_sum_line] + medium_sum_lines + electron_wave_lines + [green_dot]


# Update function
def update(frame):
    # Calculate the waves in vacuum
    vacuum_waves = [wave(x[x < L], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    vacuum_sum = sum(vacuum_waves)
    vacuum_sum_line.set_data(x[x < L], vacuum_sum)

    # Reduce opacity of vacuum wave once it crosses the line x = L/2
    if frame * c > L / 2:
        vacuum_sum_line.set_alpha(0.2)  # Reduced opacity
    else:
        vacuum_sum_line.set_alpha(1)  # Full opacity


    # Calculate the waves in the medium for each slice
    slice_width = L / (2 * num_slices)
    for slice_number in range(num_slices):
        x_slice = x[
            (x >= medium_start + slice_number * slice_width) & (x < medium_start + (slice_number + 1) * slice_width)]
        medium_waves = [medium_wave(x_slice, frame, w, c, slice_number) * weight for w, weight in
                        zip(angular_frequencies, weights)]
        medium_sum = sum(medium_waves)
        medium_sum_lines[slice_number].set_data(x_slice, medium_sum)

        electron_waves = [subtracted_wave(x_slice, frame, w, c, slice_number) * weight for w, weight in
                          zip(angular_frequencies, weights)]
        electron_sum = sum(electron_waves)
        electron_wave_lines[slice_number].set_data(x_slice, electron_sum)

    # Update the dots' positions
    if frame >= green_starts:
        green_dot.set_data((frame - green_starts) * c_reduced + L / 2,
                           1)  # c_reduced*dt is the distance covered in one timestep by the green dot
    return [vacuum_sum_line] + medium_sum_lines + electron_wave_lines + [green_dot]


# Create a toggle for animation pause
paused = False


def onClick(event):
    global paused
    if paused:
        ani.event_source.start()
        paused = False
    else:
        ani.event_source.stop()
        paused = True


fig.canvas.mpl_connect('button_press_event', onClick)

# Create a legend
ax.legend(loc="upper right")

ani = FuncAnimation(fig, update, frames=t, init_func=init, blit=True, interval=50, repeat=True)
plt.show()
