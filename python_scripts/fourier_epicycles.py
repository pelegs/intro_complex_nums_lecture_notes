import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.integrate import quad_vec


def as_polar(z):
    return np.array([np.abs(z), np.angle(z)])


def as_cartesian(radii, angles):
    return radii * np.exp(angles * 1j)


pts_spatial_complex = np.array(
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
        2 + 2j,
    ]
)
# pts_spatial_float = np.array([np.real(pts_spatial_complex), np.imag(pts_spatial_complex)])

time_list = np.linspace(0, 2 * np.pi, pts_spatial_complex.shape[0])

num_steps = 1000
time_series = np.linspace(0, 2 * np.pi, num_steps)
pts_interpolated = np.interp(time_series, time_list, pts_spatial_complex)

N = 10
freqs = np.arange(-N, N + 1, 1)
coeffs = np.zeros(2 * N + 1, dtype=np.complex128)
for i, k in enumerate(freqs):
    coeffs[i] = (
        1
        / (2 * np.pi)
        * quad_vec(
            lambda t: np.interp(t, time_list, pts_spatial_complex)
            * np.exp(-k * t * 1j),
            0,
            2 * np.pi,
            limit=100,
            full_output=1,
        )[0]
    )

circles = np.zeros(shape=(num_steps, 2 * N + 1), dtype=np.complex128)
for step, time in enumerate(time_series):
    circles[step] = coeffs * np.exp(freqs * 1j * time)
circles_centers = np.cumsum(circles, axis=1)
rad_max_sum = np.sum(np.abs(coeffs))

# Graphics
fig, ax = plt.subplots()
ax.set_title("Epicycles?")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect("equal", "box")
ax.grid()

ax.plot(np.real(pts_interpolated), np.imag(pts_interpolated), "o")
(plot_lines,) = ax.plot(np.real(circles_centers[0]), np.imag(circles_centers[0]))
(total_path,) = ax.plot(
    [np.real(circles_centers[0, -1])], [np.imag(circles_centers[0, -1])], c="red"
)
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
    total_path.set_data(
        [np.real(circles_centers[:step, -1])], [np.imag(circles_centers[:step, -1])]
    )
    return [plot_lines, circle_patches, total_path]
    # return [total_path]


animation = FuncAnimation(fig=fig, func=update_animation, frames=num_steps, interval=0)

plt.show()
