import matplotlib.pyplot as plt
import numpy as np

# Constants
GRID_RANGE = 10
GRID_SIZE = 10
TIME_END = 10
TIME_STEPS = 112
DT = TIME_END / TIME_STEPS
K = 0.5
OMEGA = 3.123
c = OMEGA / K  # speed of light
hookes_constant = 0.5
damping_constant = 1
sigma = 0.8


# Define electric field function
def original_electric_field(t, k, y, omega):
    return gaussian(t, y)


def gaussian(t, y):
    coefficient = 5.0 / (sigma * np.sqrt(2 * np.pi))
    exponential_term = np.exp(-0.5 * ((y + c * t - GRID_RANGE) / sigma) ** 2)
    return coefficient * exponential_term


def electron_field_contribution(x_e, z_e, t_e, x_p, y_p, z_p, t_p, accel_history):
    r = np.sqrt((x_e - x_p) ** 2 + y_p ** 2 + (z_e - z_p) ** 2)
    if -DT < (t_p - t_e) - r/c < DT:  # A threshold to check if they're roughly equal
        accel_scalar = accel_history.get((x_e, z_e, t_e), 0)
        contrib = - np.sqrt((x_e - x_p) ** 2 + y_p ** 2) / r * accel_scalar / r
        return 1/10 * contrib
    return 0


# Create a 2D grid in the x-z plane
x_e = [-GRID_RANGE + 2 * GRID_RANGE * i / (GRID_SIZE - 1) for i in range(GRID_SIZE)]
z_e = list(x_e)
y_e = [0 for _ in x_e]

z_previous = list(z_e)  # Initial z positions
z_velocity = [0 for _ in x_e]
z_velocity_previous = [0 for _ in x_e]

grad = 123
#x_f = list(x_e)
#z_f = list(x_e)
x_f = [0]
z_f = [0]
y_f = [-GRID_RANGE + (2 * GRID_RANGE / grad) * i for i in range(grad)]
#y_f = [-GRID_RANGE + grad * GRID_RANGE * i / (grad * GRID_SIZE - 1) for i in range(grad * GRID_SIZE)]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.ion()

accel_history = {}

for t in [TIME_END * i / TIME_STEPS for i in range(TIME_STEPS)]:
    ax.cla()

    # Record the acceleration for each electron
    accel = original_electric_field(t, K, 0, OMEGA) - hookes_constant * (z_previous[0] - z_e[0]) - damping_constant * \
            z_velocity[0]
    for x_val in x_e:
        for z_val in z_e:
            accel_history[(x_val, z_val, t)] = accel


    # Combined electric field
    for x_p_val in x_f:
        for y_p_val in y_f:
            for z_p_val in z_f:
                ef_original = original_electric_field(t, K, y_p_val, OMEGA)
                ef_due_to_electrons = 0

                for x_val in x_e:
                    for z_val in z_e:
                        for t_e in [t * i / int(t / DT) for i in range(int(t / DT))]:
                            contribution = electron_field_contribution(x_val, z_val, t_e, x_p_val, y_p_val, z_p_val, t, accel_history)
                            ef_due_to_electrons += contribution

                ef_combined = ef_original + ef_due_to_electrons
                #ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_combined, color='m', alpha=0.5)
                ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_due_to_electrons, color='g', alpha=0.5)
                ax.quiver(x_p_val, y_p_val, z_p_val, 0, 0, ef_original, color='r', alpha=0.1)

    for i in range(len(x_e)):
        for j in range(len(z_e)):
            z_previous[j] += z_velocity[j] * DT + 0.5 * accel * DT ** 2
            ax.scatter(x_e[i], y_e[i], z_previous[j], c='b', alpha=0.5)

            z_velocity[j] = z_velocity_previous[j] + accel * DT

    z_velocity_previous = z_velocity.copy()

    ax.set_xlim([-GRID_RANGE, GRID_RANGE])
    ax.set_ylim([-GRID_RANGE, GRID_RANGE])
    ax.set_zlim([-GRID_RANGE, GRID_RANGE])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.draw()
    plt.pause(0.01)

    if not plt.get_fignums():
        break