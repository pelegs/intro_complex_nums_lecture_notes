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

num_circles = 10
pts_circles = np.fft.fft(pts_spatial, n=num_circles)

circle_params = as_polar(pts_circles)
circles = np.zeros(shape=(num_steps, num_circles), dtype=np.complex128)
for step, t in enumerate(time):
    circles[step] = as_cartesian(circle_params[0], circle_params[1] * t)
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
