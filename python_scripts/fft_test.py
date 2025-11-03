import numpy as np
from scipy.integrate import quad_vec

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

time_list = np.linspace(0, 2 * np.pi, pts_spatial_complex.shape[0])
num_steps = 200
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
coeff_norms = np.abs(coeffs)
coeffs_sorted = np.argsort(coeff_norms)
# In the next two lines the coefficients and frequencies arrays are sorted by the
# absolute value of the coefficients (so that the circles shrink with index)
coeffs = coeffs[np.flip(coeffs_sorted)]
freqs = freqs[np.flip(coeffs_sorted)]

coeffs_fft = np.fft.fft(pts_spatial_complex, 2 * N + 1)
freqs_fft = np.fft.fftfreq(2 * N + 1)

# print(np.abs(coeffs))
print(coeffs_fft)
print(freqs_fft)
