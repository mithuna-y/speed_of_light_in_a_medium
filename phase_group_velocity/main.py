import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulation:
    def __init__(self):
        # Graph parameters
        self.L = 200
        self.T = 100
        self.dt = 0.1

        # Central frequency of the Gaussian.
        self.f0 = 1
        self.main_angular_frequency = 2 * np.pi * self.f0

        # Width of the Gaussian.
        self.sigma = 0.5

        # Speed of light
        self.c = 20

        # Material constants
        self.material_constant = 1000
        self.resonant_frequency = 4
        self.resonant_ang_frequency = 2 * np.pi * self.resonant_frequency

        self.frequencies = [self.f0-0.05, self.f0+0.05]
        self.weights = [1, 1]
        self.angular_frequencies = [2 * np.pi * f for f in self.frequencies]
        self.wavenumbers = [(1 + self.material_constant/ (self.resonant_ang_frequency**2 - w**2)) * w / self.c for w in self.angular_frequencies]

        # Velocity calculations
        self.group_velocity = self.c / (1 + self.material_constant * ((self.resonant_ang_frequency ** 2 + self.main_angular_frequency ** 2) / (self.resonant_ang_frequency ** 2 - self.main_angular_frequency ** 2) ** 2))
        self.phase_velocity = self.c / (1 + self.material_constant * (1 / (self.resonant_ang_frequency ** 2 - self.main_angular_frequency ** 2)))

        self.x = np.linspace(0, self.L, 1000)
        self.t = np.arange(0, self.T, self.dt)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        # Create line objects for the wave
        self.wave_lines = []
        self.wave_lines.append(self.ax.plot([], [], color="red", alpha=0.3)[0])
        self.wave_lines.append(self.ax.plot([], [], color="darkblue", alpha=0.3)[0])

        # Create a line object for the sum of waves
        self.sum_line, = self.ax.plot([], [], color="purple")

        # Create a line object for the envelope
        self.envelope_line, = self.ax.plot([], [], color="lightblue")

        # Create point objects for the group velocity and phase velocity
        self.group_velocity_point, = self.ax.plot([], [], 'bo')
        self.phase_velocity_point, = self.ax.plot([], [], 'yo')

        # Animation control
        self.is_paused = False
        self.fig.canvas.mpl_connect('button_press_event', self.onClick)

        self.ani = FuncAnimation(self.fig, self.update, frames=self.t, init_func=self.init, blit=True, interval=50, repeat=True)

    def wave(self, x, t, w, k):
        return np.cos(w * t - k * x)

    def difference_wave(self, x, t, w1, w2, k1, k2):
        return 2 * self.wave(x, t, (w2-w1)/2, (k2-k1)/2)

    def average_wave(self, x, t, w1, w2, k1, k2):
        return 2 * self.wave(x, t, (w2+w1)/2, (k2+k1)/2)

    def init(self):
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(-2, 2)
        for line in self.wave_lines + [self.sum_line, self.envelope_line, self.group_velocity_point, self.phase_velocity_point]:
            line.set_data([], [])
        return self.wave_lines + [self.sum_line, self.envelope_line, self.group_velocity_point, self.phase_velocity_point]

    def onClick(self, event):
        if self.is_paused:
            self.ani.event_source.start()
        else:
            self.ani.event_source.stop()
        self.is_paused = not self.is_paused

    def update(self, frame):
        # Calculate the waves
        waves = [self.wave(self.x, frame, w, k) * weight for w, k, weight in zip(self.angular_frequencies, self.wavenumbers, self.weights)]
        sum_of_waves = sum(waves)
        self.sum_line.set_data(self.x, sum_of_waves)

        # Plot the individual waves
        for line, wave_data in zip(self.wave_lines, waves):
            line.set_data(self.x, wave_data)

        # Calculate the envelope
        difference_wave = self.difference_wave(self.x, frame, self.angular_frequencies[0], self.angular_frequencies[1], self.wavenumbers[0], self.wavenumbers[1])
        self.envelope_line.set_data(self.x, difference_wave)

        # Update the group and phase velocity points
        self.group_velocity_point.set_data(frame * self.group_velocity, 2)
        phase_velocity_y = self.difference_wave(frame * self.phase_velocity, frame, self.angular_frequencies[0], self.angular_frequencies[1], self.wavenumbers[0], self.wavenumbers[1])
        self.phase_velocity_point.set_data(frame * self.phase_velocity, phase_velocity_y)

        return self.wave_lines + [self.sum_line, self.envelope_line, self.group_velocity_point, self.phase_velocity_point]

    def run(self):
        plt.show()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
