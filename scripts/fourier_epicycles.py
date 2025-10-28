import matplotlib.pyplot as plt
import numpy as np


def as_polar(z):
    return np.array([np.abs(z), np.angle(z)])


def as_cartesian(radii, angles):
    return radii * np.exp(angles * 1j)


pts_spatial = np.array(
    [
        1 + 1j,
        -1 + 1j,
        -1 + -1j,
        1 - 1j,
    ]
)
num_steps = 100
time = np.linspace(0, 2 * np.pi, num_steps)

N = 100
freqs = np.arange(-N, N + 1, 1).astype(int)
num_circles = freqs.shape[0]
pts_circles = np.fft.ifft(pts_spatial, n=num_circles)
num_circles = pts_circles.shape[0]
circles_array = np.zeros(shape=(num_steps, num_circles), dtype=np.complex128)
circles = np.zeros(shape=(num_steps, num_circles), dtype=np.complex128)
for step, t in enumerate(time):
    circles_array[step] = np.exp(freqs * 1j * t)
    circles[step] = pts_circles * circles_array[step]
circles_cumsums = np.cumsum(circles, axis=1)

# Graphics
fig, ax = plt.subplots()
ax.set_title("Epicycles?")
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
ax.set_aspect("equal", "box")
ax.grid()

frame_data = circles_cumsums[-1]
plt.plot(np.real(frame_data), np.imag(frame_data))

plt.show()
