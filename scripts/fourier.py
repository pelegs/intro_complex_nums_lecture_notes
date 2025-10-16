import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy import typing as npt


def as_cartesian(z_polar: npt.NDArray[np.float64]) -> npt.NDArray[np.complex64]:
    return z_polar[0] * np.exp(1j * z_polar[1])


def as_polar(z: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
    mods: npt.NDArray[np.float64] = np.abs(z).astype(np.float64)
    args: npt.NDArray[np.float64] = np.angle(z).astype(np.float64)
    both = [mods, args]
    return np.array(both)


def normalize(z: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    return z / np.abs(z)


num_basis_funcs: int = 5
basis_funcs: npt.NDArray[np.float64] = np.arange(0, num_basis_funcs, dtype=np.float64)
coeffs: npt.NDArray[np.complex64] = np.zeros(num_basis_funcs, dtype=np.complex64)
# for n in range(num_basis_funcs):
#     if n % 2 == 1:
#         coeffs[n] = (-1) ** ((n - 1) / 2) * 4.0 / (n * np.pi)
# coeffs: npt.NDArray[np.float64] = np.random.uniform(
#     low=0.0, high=2.0, size=num_basis_funcs
# )
coeffs[1] = coeffs[2] = 1.0
coeffs[3] = coeffs[4] = 0.5

num_steps: int = 360
time_series: npt.NDArray[np.float64] = np.linspace(
    0, 2 * np.pi, num_steps, dtype=np.float64
)

z_arr: npt.NDArray[np.complex64] = np.zeros(
    shape=(num_steps, num_basis_funcs), dtype=np.complex64
)


for i, t in enumerate(time_series):
    z_arr[i] = coeffs * np.exp(1.0j * basis_funcs * t)

z_sum: npt.NDArray[np.complex64] = np.sum(z_arr, axis=1)
z_sum_real: npt.NDArray[np.float64] = np.real(z_sum).astype(np.float64)
z_sum_imag: npt.NDArray[np.float64] = np.imag(z_sum).astype(np.float64)

z_vals: npt.NDArray[np.complex64] = np.cumsum(z_arr, axis=1)

# ------- #
#  Figure #
# ------- #

fig, ax = plt.subplots(1, 2, figsize=(10, 8))

ax[0].set_aspect("equal", "box")
ax[0].set_xlabel("Real")
ax[0].set_ylabel("Imaginary")
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)
(complex_lines,) = ax[0].plot(
    np.real(z_vals[0]),
    np.imag(z_vals[0]),
    color="green",
    marker="o",
    linestyle="solid",
    linewidth=1,
    markersize=2,
)
(complex_outline,) = ax[0].plot(
    np.real(z_vals[0, -1]),
    np.imag(z_vals[0, -1]),
    color="blue",
    linewidth=1,
)

ax[1].set_xlabel("Time")
ax[1].set_ylabel("Value (imaginary)")
ax[1].set_xlim(0, 2 * np.pi)
ax[1].set_ylim(-5, 5)
(imag_line,) = ax[1].plot(
    time_series[0], np.imag(z_vals[0, -1]), color="green", linewidth=1
)


# --------- #
# Animation #
# --------- #


def update_animation(frame):
    complex_lines.set_data(np.real(z_vals[frame]), np.imag(z_vals[frame]))
    complex_outline.set_data(np.real(z_vals[:frame, -1]), np.imag(z_vals[:frame, -1]))
    imag_line.set_data(time_series[:frame], np.imag(z_vals[:frame, -1]))
    return [complex_lines, complex_outline, imag_line]


animation: FuncAnimation = FuncAnimation(
    fig=fig, func=update_animation, frames=num_steps, interval=0
)

plt.show()
