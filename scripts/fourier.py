from turtle import width

import matplotlib.gridspec as gridspec
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


num_basis_funcs: int = 50
basis_funcs: npt.NDArray[np.float64] = np.arange(0, num_basis_funcs, dtype=np.float64)
coeffs: npt.NDArray[np.complex64] = np.zeros(num_basis_funcs, dtype=np.complex64)
# for n in range(num_basis_funcs):
#     if n % 2 == 1:
#         coeffs[n] = (-1) ** ((n - 1) / 2) * 4.0 / (n * np.pi)
# coeffs: npt.NDArray[np.float64] = np.random.uniform(
#     low=0.0, high=2.0, size=num_basis_funcs
# )
coeffs[1] = coeffs[2] = 1.0
coeffs[13] = coeffs[24] = 0.5

num_steps: int = 1000
time_series: npt.NDArray[np.float64] = np.linspace(
    0, 2 * np.pi, num_steps, dtype=np.float64
)

z_arr: npt.NDArray[np.complex64] = np.zeros(
    shape=(num_steps, num_basis_funcs), dtype=np.complex64
)


for i, t in enumerate(time_series):
    z_arr[i] = coeffs * np.exp(1.0j * basis_funcs * t)

# Normalizing the sequence, otherwise max values depend on coefficients
z_arr /= np.cumsum(z_arr[0])[-1]

z_sum: npt.NDArray[np.complex64] = np.sum(z_arr, axis=1)
z_sum_real: npt.NDArray[np.float64] = np.real(z_sum).astype(np.float64)
z_sum_imag: npt.NDArray[np.float64] = np.imag(z_sum).astype(np.float64)

z_vals: npt.NDArray[np.complex64] = np.cumsum(z_arr, axis=1)

# ------- #
#  Figure #
# ------- #

gs = gridspec.GridSpec(4, 4, height_ratios=[1] * 4, width_ratios=[1] * 4)

# gs.update(left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)

ax_complex = plt.subplot(gs[0, 0])
ax_real = plt.subplot(gs[1:, 0])
ax_imag = plt.subplot(gs[0, 1:])
# fig, ax = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [1, 1]})

ax_complex.set_aspect("equal")
ax_complex.set_xlabel("Real")
ax_complex.set_ylabel("Imaginary")
ax_complex.set_xlim(-1.2, 1.2)
ax_complex.set_ylim(-1.2, 1.2)
(complex_lines,) = ax_complex.plot(
    np.real(z_vals[0]),
    np.imag(z_vals[0]),
    color="green",
    marker="o",
    linestyle="solid",
    linewidth=1,
    markersize=2,
)
(complex_outline,) = ax_complex.plot(
    np.real(z_vals[0, -1]),
    np.imag(z_vals[0, -1]),
    color="blue",
    linewidth=1,
)

ax_imag.set_aspect("equal")
ax_imag.set_xlabel("Time")
ax_imag.set_ylabel("Value (imaginary)")
ax_imag.set_xlim(0, 2 * np.pi)
ax_imag.set_ylim(-1.2, 1.2)
(imag_line,) = ax_imag.plot(
    time_series[0], np.imag(z_vals[0, 0]), color="green", linewidth=1
)

ax_real.set_aspect("equal")
ax_real.set_xlabel("Value (real)")
ax_real.set_ylabel("Time")
ax_real.set_xlim(-1.2, 1.2)
ax_real.set_ylim(0, 2 * np.pi)
(real_line,) = ax_real.plot(
    np.real(z_vals[0, 0]), time_series[0], color="red", linewidth=1
)


# --------- #
# Animation #
# --------- #


def update_animation(frame):
    complex_lines.set_data(np.real(z_vals[frame]), np.imag(z_vals[frame]))
    complex_outline.set_data(np.real(z_vals[:frame, -1]), np.imag(z_vals[:frame, -1]))
    imag_line.set_data(time_series[:frame], np.imag(z_vals[:frame, -1]))
    real_line.set_data(np.real(z_vals[:frame, -1]), 2 * np.pi - time_series[:frame])
    return [complex_lines, complex_outline, imag_line, real_line]


fig = plt.gcf()
animation: FuncAnimation = FuncAnimation(
    fig=fig, func=update_animation, frames=num_steps, interval=0
)

plt.show()
