import matplotlib.pyplot as plt
import numpy as np

# Constants
TIME_END = 5
TIME_STEPS = 50
DT = TIME_END / TIME_STEPS
c = 6  # speed of light
hookes_constant = 20
mass_electron = 1
damping_constant = 0
material_y_width = 20
number_of_y_points = 100  # Increased resolution
angular_frequency = np.sqrt(hookes_constant/mass_electron) * 0.5

# Define electric field function
def plane_wave(t, y, angular_frequency):
    return np.exp(1j * (angular_frequency * (t - y / c)))

# Electron field contribution (radiative field)
def electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_scalar):
    r = np.sqrt((x_e - x_p) ** 2 + (y_e - y_p) ** 2 + (z_e - z_p) ** 2)
    if r == 0:
        return 0
    if -DT < (t_p - t_e) - r / c < DT:  # A threshold to check if they're roughly equal
        contrib = - np.sqrt((x_e - x_p) ** 2 + (y_e - y_p) ** 2) / r * accel_scalar / r
        return contrib
    return 0

# Initialize electron position and velocity
electron_position = 0
electron_velocity = 0
electron_acceleration_history = np.zeros(TIME_STEPS)

# y values where the field will be evaluated
y_f = np.linspace(-material_y_width, material_y_width, number_of_y_points)

# Plotting
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.view_init(elev=0, azim=0)

for t_index, t in enumerate(np.linspace(0, TIME_END, TIME_STEPS, endpoint=False)):
    ax.cla()

    for y_p_val in y_f:
        ef_external = plane_wave(t, y_p_val, angular_frequency)

        # Calculate radiative field due to electron motion
        ef_radiative = electron_field_contribution(0, electron_position, 0, t, 0, y_p_val, 0, t, electron_acceleration_history[t_index])

        # Combine external and radiative fields
        ef_combined = ef_external + ef_radiative

        # Plot the fields
        ax.quiver(0, y_p_val, 0, 0, 0, ef_combined.real, color='m', alpha=1)

    # Update electron position and velocity
    electron_acceleration = ef_combined.real - hookes_constant * electron_position - damping_constant * electron_velocity
    electron_velocity += electron_acceleration * DT
    electron_position += electron_velocity * DT
    electron_acceleration_history[t_index] = electron_acceleration

    ax.set_xlim([-1, 1])
    ax.set_ylim([-material_y_width, material_y_width])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.draw()
    plt.pause(0.01)

    if not plt.get_fignums():
        break
