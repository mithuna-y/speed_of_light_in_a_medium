import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

#graph parameters
L = 200
T = 100
dt = 0.1
medium_start = L / 2

# Material constants
material_constant = 50
resonant_frequency = 4
resonant_angular_frequency = 2 * np.pi * resonant_frequency

# Central frequency of the Gaussian.
f0 = 2
main_angular_frequency = 2 * np.pi * f0

# Width of the Gaussian.
sigma = 0.5

c = 20
group_velocity = c / (1 + material_constant * ((resonant_angular_frequency ** 2 + main_angular_frequency ** 2) / (resonant_angular_frequency ** 2 - main_angular_frequency ** 2) ** 2))
phase_velocity = c / (1 + material_constant * (1 / (resonant_angular_frequency ** 2 - main_angular_frequency ** 2)))
blue_starts = (L / 2) / c + 0.03
alpha = 0.1
num_slices = 100

frequencies = np.linspace(0, 2*f0, 500)
weights = 1/100 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)
angular_frequencies = [2 * np.pi * f for f in frequencies]

print(f"resonant angular frequency = {resonant_angular_frequency}, main angular frequency = {main_angular_frequency}")
refractive_index_theory = 1 + material_constant / (resonant_angular_frequency**2 - main_angular_frequency**2)
print(f"theory refractive index: {refractive_index_theory}")

x = np.linspace(0, L, 1000)
t = np.arange(0, T, dt)

fig, ax = plt.subplots(figsize=(8, 6))

# colour in the medium
for slice_number in range(num_slices):
    ax.add_patch(Rectangle((medium_start + slice_number * L / (2 * num_slices), -2), L / (2 * num_slices), 4, facecolor="lightgray", alpha=1 / (slice_number + 1)))

# Create line objects for the wave
vacuum_sum_line, = ax.plot([], [], color="purple")
medium_sum_lines = [ax.plot([], [], color="purple")[0] for _ in range(num_slices)]

# Create line objects for the envelope
vacuum_envelope_line, = ax.plot([], [], color="lightblue")
medium_envelope_lines = [ax.plot([], [], color="lightblue")[0] for _ in range(num_slices)]

# Dots that move at various speeds
red_dot = ax.scatter([], [], color='red', zorder=10)
blue_dot = ax.scatter([], [], color='blue', zorder=10)
yellow_dot = ax.scatter([], [], color='yellow', zorder=10)


def wave(x, t, w, c):
    return np.cos(w * (t - x/c))


def medium_wave(x, t, w, c, slice_number):
    slice_width = (L/2)/num_slices
    phase = material_constant * (slice_number+1) * w / (
                resonant_angular_frequency ** 2 - w ** 2 + 0.012)*slice_width/c
    return np.cos(w * (t - x/c) - phase)


# Define the complex wave functions
def complex_wave(x, t, w, c):
    return np.exp(1j * w * (t - x/c))


def complex_medium_wave(x, t, w, c, slice_number):
    slice_width = (L/2)/num_slices
    phase = material_constant * (slice_number+1) * w / (
                resonant_angular_frequency ** 2 - w ** 2 + 0.012)*slice_width/c
    return np.exp(1j * (w * (t - x/c) - phase))

def init():
    ax.set_xlim(0, L)
    ax.set_ylim(-2, 2)
    for line in [vacuum_sum_line, vacuum_envelope_line] + medium_sum_lines + medium_envelope_lines:
        line.set_data([], [])
    red_dot.set_offsets([0, 0])
    blue_dot.set_offsets([-100, 0])
    yellow_dot.set_offsets([-100, 0])
    return [vacuum_sum_line, vacuum_envelope_line] + medium_sum_lines + medium_envelope_lines + [red_dot, blue_dot, yellow_dot]

def update(frame):
    # Calculate the waves in vacuum
    vacuum_waves = [wave(x[x < medium_start], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    vacuum_sum = sum(vacuum_waves)
    vacuum_sum_line.set_data(x[x < medium_start], vacuum_sum)

    # Calculate the envelope in vacuum
    vacuum_complex_waves = [complex_wave(x[x < medium_start], frame, w, c) * weight for w, weight in zip(angular_frequencies, weights)]
    vacuum_envelope = np.abs(sum(vacuum_complex_waves))
    vacuum_envelope_line.set_data(x[x < medium_start], vacuum_envelope)

    # Calculate the waves in the medium for each slice
    slice_width = L / (2 * num_slices)
    medium_envelope_array = []  # An array to hold the envelope for each slice
    for slice_number in range(num_slices):
        x_slice = x[(x >= medium_start + slice_number * slice_width) & (x < medium_start + (slice_number + 1) * slice_width)]
        medium_waves = [medium_wave(x_slice, frame, w, c, slice_number) * weight for w, weight in zip(angular_frequencies, weights)]
        medium_sum = sum(medium_waves)
        medium_sum_lines[slice_number].set_data(x_slice, medium_sum)

        # Calculate the envelope in the medium
        medium_complex_waves = [complex_medium_wave(x_slice, frame, w, c, slice_number) * weight for w, weight in zip(angular_frequencies, weights)]
        medium_envelope = np.abs(sum(medium_complex_waves))
        medium_envelope_array.append(medium_envelope)
        medium_envelope_lines[slice_number].set_data(x_slice, medium_envelope)

    # Update the position of the dots
    if frame >= blue_starts:
        blue_x = (frame-blue_starts)*group_velocity + L/2
        slice_number = min(int((blue_x - L / 2) // slice_width), num_slices - 1)
        x_slice = x[(x >= medium_start + slice_number * slice_width) & (x < medium_start + (slice_number + 1) * slice_width)]
        blue_y = np.interp(blue_x, x_slice, medium_envelope_array[slice_number])
        blue_dot.set_offsets([[blue_x, blue_y]])

        yellow_x = (frame-blue_starts)*phase_velocity + L/2
        slice_number = min(int((yellow_x - L / 2) // slice_width), num_slices - 1)
        x_slice = x[(x >= medium_start + slice_number * slice_width) & (x < medium_start + (slice_number + 1) * slice_width)]
        yellow_y = np.interp(yellow_x, x_slice, medium_envelope_array[slice_number])
        yellow_dot.set_offsets([[yellow_x, yellow_y]])

    red_x = frame * c
    if red_x < L / 2:  # The red dot is in vacuum.
        red_y = np.interp(red_x, x[x < medium_start], vacuum_envelope)
    else:  # The red dot is in the medium.
        if 'red_y_initial' not in update.__dict__:
            red_y = blue_y
    red_dot.set_offsets([[red_x, red_y]])



    return [vacuum_sum_line, vacuum_envelope_line] + medium_sum_lines + medium_envelope_lines + [red_dot, blue_dot, yellow_dot]


ani = FuncAnimation(fig, update, frames=t, init_func=init, blit=True, interval=10, repeat=True)

plt.show()
