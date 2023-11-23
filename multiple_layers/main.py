import matplotlib.pyplot as plt
import numpy as np

# Constants
material_y_width = 20
material_x_z_width = 10
number_of_electrons_wide = 3
number_of_layers = 4
number_of_y_points = number_of_layers * 40
TIME_END = 200
TIME_STEPS = 500
DT = TIME_END / TIME_STEPS
c = 6  # speed of light
hookes_constant = 1
mass_electron = 1
damping_constant = 1
sigma = 0.8
resonant_angular_frequency = np.sqrt(hookes_constant/mass_electron)
angular_frequency = resonant_angular_frequency * 0.5

# Define electric field function
def original_electric_field(t, y, angular_frequency):
    return plane_wave(t, y, angular_frequency)


def gaussian(t, y):
    coefficient = 5.0 / (sigma * np.sqrt(2 * np.pi))
    exponential_term = np.exp(-0.5 * ((y + c * t - material_y_width/4) / sigma) ** 2)
    return coefficient * exponential_term


def plane_wave(t, y, angular_frequency):
    return np.exp(1j*(angular_frequency*(t + y/c)))


def electron_field_contribution(x_e, y_e_layer, z_e, t_e, x_p, y_p, z_p, t_p, accel_hist):
    r = np.sqrt((x_e - x_p) ** 2 + (y_e_layer - y_p) ** 2 + (z_e - z_p) ** 2)
    if r == 0:
        return 0
    assert t_p >= t_e, f't_p should be less than t_e, but got t_p {t_p} and t_e {t_e}'
    if -DT < (t_p - t_e) - r / c < DT:  # A threshold to check if they're roughly equal
        index = int(t_e / DT)
        accel_scalar = accel_hist[index]
        contrib = - np.sqrt((x_e - x_p) ** 2 + (y_e_layer - y_p) ** 2) / r * accel_scalar / r
        return 1/(number_of_electrons_wide**2) * contrib
    return 0


def layer_y_position(layer_number, number_of_layers):
    return - (layer_number-1) * material_y_width / number_of_layers


def force(ef_combined, z_previous, layer, z_velocity):
    return ef_combined - hookes_constant * z_previous[layer] - damping_constant * z_velocity[layer]




# Initialize arrays for electron positions and velocities
x_e = np.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide, endpoint=True)
z_e = np.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide, endpoint=True)

y_e = np.array([layer_y_position(i + 1, number_of_layers) for i in range(number_of_layers)])

# y values where the field will be evaluated.
y_f = [-material_y_width + (2 * material_y_width / number_of_y_points) * i for i in range(number_of_y_points)]


z_middle_of_layer = np.zeros_like(y_e)
z_velocity_middle_of_layer = np.zeros_like(z_middle_of_layer)
z_velocity_previous_middle_of_layer = np.zeros_like(z_middle_of_layer)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0, azim=0)

plt.ion()

accel_history = np.zeros((number_of_layers, TIME_STEPS))

for step in range(TIME_STEPS):
    ax.cla()
    t = step * DT

    for y_p_val in y_f:
        x_p_val = 0
        z_p_val = 0

        ef_original = original_electric_field(t, y_p_val, angular_frequency)
        ef_due_to_electrons = 0

        for x_electron in x_e:
            for z_electron in z_e:
                for t_e_step in range(step):
                    t_e = t_e_step * DT
                    contribution = 0
                    for layer in range(number_of_layers):
                        y_e_layer = layer_y_position(layer, number_of_layers)
                        contribution += electron_field_contribution(x_electron, y_e_layer, z_electron, t_e, x_p_val, y_p_val, z_p_val, t, accel_history[layer])
                    ef_due_to_electrons += contribution

        ef_combined = ef_original + ef_due_to_electrons

        for layer in range(number_of_layers):
            if (x_p_val, y_p_val, z_p_val) == (0, layer_y_position(layer, number_of_layers), 0):
                accel_history[layer, step] = force(ef_combined, z_middle_of_layer, layer, z_velocity_middle_of_layer) / mass_electron

        #ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_combined, color='m', alpha=1)
        ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_original, color='r', alpha=0.2)
        ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_due_to_electrons * 5, color='b', alpha=0.2)

    for layer in range(number_of_layers):
        z_middle_of_layer[layer] += z_velocity_middle_of_layer[layer] * DT + 0.5 * accel_history[layer, step] * DT ** 2
        z_velocity_middle_of_layer[layer] = z_velocity_previous_middle_of_layer[layer] + accel_history[layer, step] * DT

        z_velocity_previous_middle_of_layer[layer] = z_velocity_middle_of_layer[layer]

        for x in x_e:
            for z in z_e:
                ax.scatter(x, y_e[layer], z_middle_of_layer[layer] + z, c='b', alpha=0.5)

    ax.set_xlim([-material_x_z_width, material_x_z_width])
    ax.set_ylim([-material_y_width, material_y_width / 2])
    ax.set_zlim([-material_x_z_width, material_x_z_width])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.draw()
    plt.pause(0.01)

    if not plt.get_fignums():
        break