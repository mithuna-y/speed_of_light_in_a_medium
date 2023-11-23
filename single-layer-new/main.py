import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define electric field function
def electric_field(t, k, y, omega):
    return np.cos(omega * t - k * y)



# Define initial electron positions on a 2D grid in the x-z plane
n = 5  # Grid size for electrons
x_e, z_e = np.linspace(-5, 5, n), np.linspace(-5, 5, n)
x_e, z_e = np.meshgrid(x_e, z_e)
y_e = np.zeros_like(x_e)
z_previous = np.copy(z_e)  # Initialize z_previous to be the same as initial z_e


# Define 3D grid for plotting electric field
x_f, y_f, z_f = np.linspace(-5, 5, n), np.linspace(-5, 5, 2*n), np.linspace(-5, 5, n)
x_f, y_f, z_f = np.meshgrid(x_f, y_f, z_f)

# Constants
k = 1
omega = 1
amplitude = 0.5

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Turn on interactive mode
plt.ion()

# Initialize dictionary to store acceleration history for each electron
accel_history = {}

# Time loop
for t in np.linspace(0, 100, 1000):
    dt = 10/100

    # Clear previous plot
    ax.clear()

    # Plot electric field as arrows in 3D space
    ax.quiver(x_f, y_f, z_f, 0, 0, electric_field(t, k, y_f, omega), color='b',
              label='Electric Field')

    # Calculate the acceleration (assuming F = ma, and m=1 for simplicity)
    accel = electric_field(t, k, y_e, omega)

    # Record the acceleration for each electron
    for x_val, y_val, z_val in zip(x_e.ravel(), y_e.ravel(), z_e.ravel()):
        key_tuple = (x_val, y_val, t)
        key_str = str(key_tuple)

        if key_str not in accel_history:
            accel_history[key_str] = []

        accel_history[key_str].append(accel)

    # Update electron positions based on the electric field
    delta_z = accel * dt

    # Scatter plot of electrons at new positions
    ax.scatter(x_e, y_e, z_previous + delta_z, c='r', label='Electrons')

    # Update z_previous for the next iteration
    z_previous = z_previous + dt * delta_z

    # Set plot properties
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.draw()
    plt.pause(0.001)

# Turn off interactive mode
plt.ioff()

# Show plot
plt.show()
