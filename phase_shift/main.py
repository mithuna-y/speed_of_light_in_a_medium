import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Constants
w = 2*np.pi*1.5 # Angular frequency
c = 3  # Speed of wave
n = 1.5  # Refractive index

# Time and space variables
t_values = np.linspace(0, 10, 100)
z_values = np.linspace(0, 10, 100)

# Create figure and axis
fig, ax = plt.subplots()
line1, = ax.plot([], [], lw=2, label='with n')
line2, = ax.plot([], [], lw=2, label='normal')
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.legend()

paused = False

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return (line1, line2,)

def update(frame):
    if not paused:
        t = frame
        y1 = np.cos(w * t - (1 / (c) * (z_values+0.5)))
        y2 = np.cos(w * t - 1 / (c) * z_values)
        line1.set_data(z_values, y1)
        line2.set_data(z_values, y2)
    return (line1, line2,)

def onClick(event):
    global paused
    paused ^= True

ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 1000),
                    init_func=init, blit=True)

fig.canvas.mpl_connect('button_press_event', onClick)

plt.show()