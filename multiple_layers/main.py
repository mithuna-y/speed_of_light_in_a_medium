import matplotlib.pyplot as plt
import numpy as np

# Constants
material_y_width = 200
material_x_z_width = 10
number_of_electrons_wide = 7  # please make odd, thank you
number_of_layers = 10
charge_electron = 1 / number_of_electrons_wide ** 2  # this is to ensure the amount of charge per layer is constant
TIME_END = 200
TIME_STEPS = 200
DT = TIME_END / TIME_STEPS
start_y = material_y_width * 1 / 8  # this is where we will start plotting y from
number_of_evaluation_points = 60  # make this larger for more resolution in the y axis
c = 6  # speed of light
hookes_constant = 0.1
mass_electron = 1
damping_constant = 1
sigma = 10
resonant_angular_frequency = np.sqrt(hookes_constant / mass_electron)
angular_frequency = resonant_angular_frequency * 1.5


# Define electric field function
def original_electric_field(t, y):
    return plane_wave(t, y, angular_frequency)
    #return gaussian(t, y)


def gaussian(t, y):
    coefficient = 100 / (sigma * np.sqrt(2 * np.pi))
    exponential_term = np.exp(-0.5 * ((y + c * t - material_y_width / 4) / sigma) ** 2)
    return coefficient * exponential_term


def plane_wave(t, y, angular_frequency):
    return 10 * np.cos((angular_frequency * (t + y / c)))

# The electric field at a point p due to an electron at position x_e, y_e, z_e, t_e.
def electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_hist):
    r = np.sqrt((x_e - x_p) ** 2 + (y_e - y_p) ** 2 + (z_e - z_p) ** 2)
    if r == 0:  # This is to prevent a point from experiences a field due to an electron located there.
        return 0
    # Ensure that the point is at a later time than the electron
    assert t_p >= t_e, f't_p should be less than t_e, but got t_p {t_p} and t_e {t_e}'

    # If (t_p - t_e)  is approximately r / c, this electron is currently contributing to the electric field of this point
    if -DT < (t_p - t_e) - r / c < DT:
        index = int(t_e / DT)
        accel_scalar = accel_hist[index]
        # The size of the contribution is given in the Feynman lectures, vol 1, eq 29-1
        # https://www.feynmanlectures.caltech.edu/I_29.html
        contrib = - np.sqrt((x_e - x_p) ** 2 + (y_e - y_p) ** 2) / r * accel_scalar / r
        return charge_electron * contrib
    return 0


def force(total_electric_field, z, z_velocity):
    # Calculates the force the electron in the middle of the layer experiences
    # z is the previous z positions of the electrons in the middle of each layer
    return total_electric_field - hookes_constant * z - damping_constant * z_velocity


def set_electron_y_positions(material_y_width, number_of_layers):
    def layer_y_position(layer_number, number_of_layers):
        return - layer_number * material_y_width / number_of_layers

    return np.array([layer_y_position(i, number_of_layers) for i in range(number_of_layers)])


# This function determines where the electrons are in each layer (x_electrons, z_electrons)
# and where the layers are (y_electrons)
def set_electron_xz_positions(material_x_z_width, number_of_electrons_wide):
    if number_of_electrons_wide == 1:
        # Single electron at the center
        x_e = np.array([0])
        z_e = np.array([0])
    else:
        # Create a grid of electron positions
        x_e = np.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide, endpoint=True)
        z_e = np.linspace(-material_x_z_width, material_x_z_width, number_of_electrons_wide, endpoint=True)

    return x_e, z_e



# total electric field due to all electrons in the past on a point (0, y, 0) at time_step
def electrons_electric_field(time_step, y):
    t = time_step * DT
    ef_due_to_electrons = 0

    # Calculate the electric field due to all electrons (at all possible t_electron) on the point (0, y_point, 0)
    for (x_electron, y_electron, z_electron) in electron_positions:
        for t_e_step in range(step):
            electron_layer = np.where(electron_y_positions == y_electron)[0][0]
            t_electron = t_e_step * DT
            contribution = 0
            contribution += electron_field_contribution(x_electron, y_electron, z_electron, t_electron,
                                                        0, y, 0, t, accel_history[electron_layer])
            ef_due_to_electrons += contribution
    return ef_due_to_electrons

if __name__ == '__main__':
    electron_y_positions = set_electron_y_positions(material_y_width, number_of_layers)
    electron_x_positions, electron_z_positions = set_electron_xz_positions(material_x_z_width, number_of_electrons_wide)
    electron_positions = [(x, y, z) for x in electron_x_positions for y in electron_y_positions for z in
                          electron_z_positions]

    # We are only going to evaluate the field along the line x = 0, z = 0. These are the y values for those points
    y_f = np.linspace(-material_y_width, start_y, number_of_evaluation_points)

    # These keep track of the z displacement, velocity and acceleration of the electrons for all previous times.
    # Initialised to contain 0s
    # We assume all electrons in the slice have the same values for these for simplicity
    z_middle_of_layer = np.zeros_like(electron_y_positions)
    z_velocity_middle_of_layer = np.zeros_like(z_middle_of_layer)
    accel_history = np.zeros((number_of_layers, TIME_STEPS))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)

    plt.ion()

    for step in range(TIME_STEPS):
        ax.cla()
        t = step * DT

        # We are going to calculate the field along the line x = 0, z = 0 and plot it
        for y_point in y_f:
            ef_original = original_electric_field(t, y_point)
            ef_due_to_electrons = electrons_electric_field(step, y_point)
            ef_combined = ef_original + ef_due_to_electrons

            ax.quiver(0, y_point, 0, 0, 0, ef_combined, color='m', alpha=0.3)
            ax.quiver(0, y_point, 0, 0, 0, ef_original, color='g', alpha=0.1)
            #ax.quiver(0, y_point, 0, 0, 0, ef_due_to_electrons * 5, color='b', alpha=0.2)

        # Now we update the z position and velocity of all our electrons.
        for layer, layer_y in np.ndenumerate(electron_y_positions):
            # To simplify, we will assume that each layer experiences the same electric field: the ef at (0, layer_y, 0)
            ef_on_layer = electrons_electric_field(step, layer_y) + original_electric_field(t, layer_y)

            # calculate the acceleration of the middle point of each layer and append the value to accel_history
            accel_history[layer, step] = force(ef_on_layer, z_middle_of_layer[layer],
                                               z_velocity_middle_of_layer[layer]) / mass_electron

            z_middle_of_layer[layer] += z_velocity_middle_of_layer[layer] * DT + 0.5 * accel_history[layer, step] * DT ** 2
            z_velocity_middle_of_layer[layer] += accel_history[layer, step] * DT

            #ax.scatter(electron_x_positions, layer_y, z_middle_of_layer[layer] + electron_z_positions, c='b', alpha=0.5)

        ax.set_xlim([-material_x_z_width, material_x_z_width])
        ax.set_ylim([-material_y_width, start_y])
        ax.set_zlim([-material_x_z_width, material_x_z_width])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.draw()
        plt.pause(0.01)

        if not plt.get_fignums():
            break
