import matplotlib.pyplot as plt
import numpy as np
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


num_basis_funcs: int = 100
basis_funcs: npt.NDArray[np.float64] = np.arange(0, num_basis_funcs, dtype=np.float64)
coeffs: npt.NDArray[np.complex64] = np.zeros(num_basis_funcs, dtype=np.complex64)
for n in range(num_basis_funcs):
    if n % 2 == 1:
        coeffs[n] = (-1) ** ((n - 1) / 2) * 4.0 / (n * np.pi)
# coeffs: npt.NDArray[np.float64] = np.random.uniform(
#     low=-1.0, high=1.0, size=num_basis_funcs
# )

num_steps: int = 360
time_series: npt.NDArray[np.float64] = np.linspace(
    0, 2 * np.pi, num_steps, dtype=np.float64
)

z_arr: npt.NDArray[np.complex64] = np.zeros(
    shape=(num_steps, num_basis_funcs), dtype=np.complex64
)


for i, t in enumerate(time_series):
    z_arr[i] = coeffs * np.exp(1.0j * basis_funcs * t)

z_sum: np.complex64 = np.sum(z_arr, axis=1)
z_sum_real: npt.NDArray[np.float64] = np.real(z_sum).astype(np.float64)
z_sum_imag: npt.NDArray[np.float64] = np.imag(z_sum).astype(np.float64)


# ------- #
#  Figure #
# ------- #

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.plot(time_series, z_sum_real)
plt.show()
