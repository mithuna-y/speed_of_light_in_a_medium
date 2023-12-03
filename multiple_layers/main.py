import itertools
import matplotlib.pyplot as plt
import numpy as np

# Constants
material_y_width = 100
material_x_z_width = 10
number_of_electrons_wide = 5
number_of_layers = 7
charge_electron = 1 / number_of_electrons_wide ** 2  # this is to ensure the amount of charge per layer is constant
TIME_END = 100
TIME_STEPS = 100
DT = TIME_END / TIME_STEPS
start_y = 25 # this is where we will start plotting y from
number_of_evaluation_points = 100  # make this larger for more resolution in the y axis
c = 5  # speed of light
hookes_constant = 0.1
mass_electron = 1
damping_constant = 0
sigma = 5
resonant_angular_frequency = np.sqrt(hookes_constant / mass_electron)
angular_frequency = resonant_angular_frequency * 0.99


# Define electric field function
def original_electric_field(t, y):
    #return plane_wave(t, y, angular_frequency)
    return gaussian(t, y)


def gaussian(t, y):
    coefficient = 50 / (sigma * np.sqrt(2 * np.pi))
    exponential_term = np.exp(-0.5 * ((y + c * t - material_y_width / 4) / sigma) ** 2)
    return coefficient * exponential_term


def plane_wave(t, y, angular_frequency):
    return 2 * np.cos((angular_frequency * (t + y / c)))

# The electric field at a point p due to an electron at position x_e, y_e, z_e, t_e.
def electron_field_contribution(x_e, y_e, z_e, t_e, x_p, y_p, z_p, t_p):
    # axes: all electrons X all positions
    x_square_diff = (x_e[:, np.newaxis] - x_p[np.newaxis, :]) ** 2
    y_square_diff = (y_e[:, np.newaxis] - y_p[np.newaxis, :]) ** 2
    z_square_diff = (z_e[:, np.newaxis] - z_p[np.newaxis, :]) ** 2
    r = np.sqrt(x_square_diff + y_square_diff + z_square_diff)
    # expanding to: all electrons X all positions X all times (1 -> broadcasted)
    x_square_diff = x_square_diff[:,:, np.newaxis]
    y_square_diff = y_square_diff[:, :, np.newaxis]
    z_square_diff = z_square_diff[:, :, np.newaxis]
    r = r[:,:,np.newaxis]

    electron_layer_idx = np.argmax(y_e[:, np.newaxis] == np.array(electron_y_positions)[np.newaxis, :],1)

    # Ensure that the point is at a later time than the electron
    assert np.all(t_p >= t_e), f't_p should be less than t_e, but got t_p {t_p} and t_e {t_e}'

    # If (t_p - t_e)  is approximately r / c,
    # this electron is currently contributing to the electric field of this point
    accel_hist_t_idx = np.uint32(t_e/DT)  # axes: all times
    # The size of the contribution is given in the Feynman lectures, vol 1, eq 29-1
    # https://www.feynmanlectures.caltech.edu/I_29.html
    # axes: all electrons X all positions X all earlier times
    contribution_if_contributing = (
       - np.sqrt(x_square_diff + y_square_diff) / r
       * accel_history[electron_layer_idx,:][:,accel_hist_t_idx][:, np.newaxis,:] / r
    )
    # we want to return 0 when r=0 to prevent electrons affecting themselves
    contribution_if_contributing[np.broadcast_to(r, contribution_if_contributing.shape) == 0] = 0
    # They contribute if the time from the time step to the position is roughly in accordance to c
    time_vs_distance_diff = (t_p - t_e[np.newaxis, np.newaxis, :]) - r / c
    contributing = (-DT < time_vs_distance_diff) & (time_vs_distance_diff < DT)
    # mask to zero those values under which the contribution is not valid
    contribution_if_contributing[~contributing] = 0

    contribution_if_contributing *= charge_electron

    return contribution_if_contributing


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
    if time_step == 0:
        # this early exit circumvents troubles with 0 sized arrays
        return np.zeros_like(y)
    t = time_step * DT

    electron_positions_np = np.array(electron_positions)
    electron_y_positions_np = np.array(electron_y_positions)
    # Calculate the electric field due to all electrons (at all possible t_electron) on the point (0, y_point, 0)

    t_electron = np.arange(time_step) * DT

    contributions = electron_field_contribution(
        electron_positions_np[:,0],
        electron_positions_np[:,1],
        electron_positions_np[:,2],
        t_electron,
        np.array([0]),
        np.array(y),
        np.array([0]),
        t,
    )
    return contributions.sum(0).sum(1)  # sum across all electrons and times


if __name__ == '__main__':
    electron_y_positions = np.array(set_electron_y_positions(material_y_width, number_of_layers))
    electron_x_positions, electron_z_positions = set_electron_xz_positions(material_x_z_width, number_of_electrons_wide)
    electron_x_positions = np.array(electron_x_positions)
    electron_z_positions = np.array(electron_z_positions)
    electron_positions = np.array(list(itertools.product(electron_x_positions, electron_y_positions, electron_z_positions)))

    # We are only going to evaluate the field along the line x = 0, z = 0. These are the y values for those points
    y_f = np.linspace(-material_y_width, start_y, number_of_evaluation_points)

    # These keep track of the z displacement, velocity and acceleration of the electrons for all previous times.
    # Initialised to contain 0s
    # We assume all electrons in the slice have the same values for these for simplicity
    z_middle_of_layer = np.zeros_like(electron_y_positions)
    z_velocity_middle_of_layer = np.zeros_like(z_middle_of_layer)
    accel_history = np.zeros((number_of_layers, TIME_STEPS))

    t_points = (np.arange(TIME_STEPS) * DT)
    # We are going to calculate the field along the line x = 0, z = 0 and plot it
    ef_original_on_evaluation_points = original_electric_field(
        t_points[:, np.newaxis],
        y_f[np.newaxis, :],
    )  # axes: time X y
    ef_original_on_electrons = original_electric_field(
        t_points[:, np.newaxis],
        electron_y_positions[np.newaxis, :],
    )  # axes: time X y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)

    plt.ion()

    for step in range(TIME_STEPS):
        ax.cla()
        t = step * DT

        # We are going to calculate the field along the line x = 0, z = 0 and plot it
        ef_due_to_electrons = electrons_electric_field(step, y_f)
        ef_combined = ef_original_on_evaluation_points[step,:] + ef_due_to_electrons
        ax.scatter(0, y_f, ef_combined, color='m', alpha=1, s=1)
        ax.scatter(0, y_f, ef_original_on_evaluation_points[step, :], color='r', alpha=1, s=1)
        ax.scatter(0, y_f, ef_due_to_electrons, color='b', alpha=1, s=1)

        # Now we update the z position and velocity of all our electrons.
        # To simplify, we will assume that each layer experiences the same electric field: the ef at (0, layer_y, 0)
        ef = electrons_electric_field(step, np.array(electron_y_positions)) + original_electric_field(t, electron_y_positions)
        # calculate the acceleration of the middle point of each layer and append the value to accel_history
        accel_history[:, step] = force(ef, z_middle_of_layer, z_velocity_middle_of_layer) / mass_electron

        z_middle_of_layer += z_velocity_middle_of_layer * DT + 0.5 * accel_history[:, step] * DT ** 2
        z_velocity_middle_of_layer += accel_history[:, step] * DT

        scatter_shape = (len(electron_x_positions), len(electron_y_positions))
        ax.scatter(
            np.broadcast_to(electron_x_positions[:, np.newaxis], scatter_shape),
            np.broadcast_to(electron_y_positions[np.newaxis,:], scatter_shape),
            z_middle_of_layer[np.newaxis,:] + electron_z_positions[:, np.newaxis],
            c='b',
            alpha=0.5,
        )

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
