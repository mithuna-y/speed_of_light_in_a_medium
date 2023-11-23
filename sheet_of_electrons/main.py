import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

e = 1
m_e = 1
epsilon_0 = 0.1
c = 3
f0 = 1
sigma =0.1

# List of angular frequencies and amplitudes
frequencies = np.linspace(0, 2*f0, 500)
weights = 1/10 * np.exp(-0.5 * ((frequencies - f0) / sigma) ** 2)
angular_frequencies = [2 * np.pi * f for f in frequencies]
dt = 1
num_electrons = 3

y_positions = np.linspace(-5, 5, num_electrons)
z_positions = np.linspace(-5, 5, num_electrons)
y_positions, z_positions = np.meshgrid(y_positions, z_positions)
x_positions = np.zeros_like(y_positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
electron_scatter = ax.scatter(x_positions, y_positions, z_positions, color='blue', s=100)

x_range = 100

ax.set_xlim(-x_range, x_range)
ax.set_ylim(-7, 7)
ax.set_zlim(-7, 7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

quivers = []


def update(frame):
    global quivers
    t = frame * dt

    while quivers:
        quivers.pop().remove()

    for x in np.linspace(-x_range, x_range, 200):
        for y, z in zip(y_positions.flatten(), z_positions.flatten()):

            # Sum contributions from all sinusoidal waves for external field
            w_vector_external = sum([E * np.cos(w * t - (w / c) * x) for w, E in zip(angular_frequencies, weights)])
            quivers.append(
                ax.quiver(x, y, z, 0, 0, w_vector_external, color='red')
            )

            # Calculate the induced field
            r = np.array([x, y, z]) - np.array([0, y, z])
            r_magnitude = np.linalg.norm(r)

            # Sum contributions from all sinusoidal waves for induced field
            if r_magnitude > 0.2:
                induced_field_magnitude = sum([
                    E * (-4 * np.pi * epsilon_0 * m_e * w ** 2 / r_magnitude ** 3) * (
                                2.0 * np.pi * w) ** 2 * e ** 2 * E * np.cos(
                        w * (t - r_magnitude / c) - w * x
                    ) for w, E in zip(angular_frequencies, weights)
                ])

                quivers.append(
                    ax.quiver(x, y, z, 0, 0, induced_field_magnitude, length=0.5, normalize=True, color='lightblue')
                )


ani = FuncAnimation(fig, update, frames=np.arange(0, 100, dt), interval=100)
plt.show()
