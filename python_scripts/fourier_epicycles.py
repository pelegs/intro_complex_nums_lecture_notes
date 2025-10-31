import enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def as_polar(z):
    return np.array([np.abs(z), np.angle(z)])


def as_cartesian(radii, angles):
    return radii * np.exp(angles * 1j)


pts_spatial = np.array(
    [
        2 + 2j,
        2 + 1.5j,
        2 + 1j,
        2 + 0.5j,
        2 + 0j,
        2 - 0.5j,
        2 - 1j,
        2 - 1.5j,
        2 - 2j,
        1.5 - 2j,
        1 - 2j,
        0.5 - 2j,
        0 - 2j,
        -0.5 - 2j,
        -1 - 2j,
        -1.5 - 2j,
        -2 - 2j,
        -2 - 1.5j,
        -2 - 1j,
        -2 - 0.5j,
        -2 + 0j,
        -2 + 0.5j,
        -2 + 1j,
        -2 + 1.5j,
        -2 + 2j,
        -1.5 + 2j,
        -1 + 2j,
        -0.5 + 2j,
        0 + 2j,
        0.5 + 2j,
        1 + 2j,
        1.5 + 2j,
    ]
)

num_steps = 100
time_series = np.linspace(0, 2 * np.pi, num_steps)
N = 2
num_spatial_pts = pts_spatial.shape[0]
basis_freqs = np.arange(-N, N + 1, 1) * 1j
# coeffs = np.fft.fft(pts_spatial, 2 * N + 1)
coeffs = np.ones(2 * N + 1)
circles = np.zeros(shape=(num_steps, 2 * N + 1), dtype=np.complex128)
for step, time in enumerate(time_series):
    circles[step] = coeffs * np.exp(basis_freqs * time)
circles_centers = np.cumsum(circles, axis=1, dtype=np.complex128)

# Graphics
fig, ax = plt.subplots()
ax.set_title("Epicycles?")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect("equal", "box")
ax.grid()

plot_lines = []
for i, circle in enumerate(circles[0]):
    plot_lines.append(
        plt.plot(
            [0, np.real(circle)],
            [0, np.imag(circle)],
        )[0]
    )


def update_animation(step):
    for center, circle, plot_line in zip(
        circles_centers[step], circles[step], plot_lines
    ):
        plot_line.set_data(
            [np.real(center), np.real(center + circle)],
            [np.imag(center), np.imag(center + circle)],
        )
    return [plot_lines]


animation = FuncAnimation(fig=fig, func=update_animation, frames=num_steps, interval=0)

plt.show()
