import enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


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

num_steps = 500
time_series = np.linspace(0, 2 * np.pi, num_steps)
N = 5
num_spatial_pts = pts_spatial.shape[0]
basis_freqs = np.arange(0, 2 * N + 1, 1) * 1j
# coeffs = np.flip(np.sort(np.random.uniform(1, 23, size=2 * N + 1)))
coeffs = np.fft.ifft(pts_spatial, 2 * N + 1)
circles = np.zeros(shape=(num_steps, 2 * N + 1), dtype=np.complex128)
for step, time in enumerate(time_series):
    circles[step] = coeffs * np.exp(basis_freqs * time)
circles_centers = np.cumsum(circles, axis=1)
rad_max_sum = np.sum(np.abs(coeffs))

# Graphics
fig, ax = plt.subplots()
ax.set_title("Epicycles?")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")
ax.set_xlim(-2 * rad_max_sum, 2 * rad_max_sum)
ax.set_ylim(-2 * rad_max_sum, 2 * rad_max_sum)
ax.set_aspect("equal", "box")
ax.grid()

(plot_lines,) = plt.plot(np.real(circles_centers[0]), np.imag(circles_centers[0]))
circle_patches = []
for circle, coeff in zip(circles_centers[0], coeffs):
    circle_patch = Circle(
        xy=(np.real(circle), np.imag(circle)), radius=np.abs(coeff), fill=False
    )
    circle_patches.append(circle_patch)
    ax.add_patch(circle_patch)


def update_animation(step):
    plot_lines.set_data(np.real(circles_centers[step]), np.imag(circles_centers[step]))
    for circle, circle_patch in zip(circles_centers[step], circle_patches):
        circle_patch.center = [np.real(circle), np.imag(circle)]
    return [plot_lines, circle_patches]


animation = FuncAnimation(fig=fig, func=update_animation, frames=num_steps, interval=0)

plt.show()
