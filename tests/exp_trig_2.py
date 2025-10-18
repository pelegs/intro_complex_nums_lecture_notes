import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# TODO: LaTeXify! Make everything parametrizeable

ax = plt.figure().add_subplot(projection="3d")

K = 2
num_samples = 60 * K
t = np.linspace(0, 2 * K * np.pi, num_samples)
w = np.exp((-0.0 + 1j) * t)
real_vals = np.real(w)
imag_vals = np.imag(w)

real_zs = np.ones(num_samples) * -1
imag_zs = np.ones(num_samples)

(exp_plot,) = ax.plot(xs=real_vals, ys=t, zs=imag_vals, label="exp(it)", linewidth=3)
(real_plot,) = ax.plot(real_vals, t, zs=-1, label="cos(t)", linewidth=2)
(imag_plot,) = ax.plot(t, imag_vals, zdir="x", zs=1, label="sin(t)", linewidth=2)

ax.set_aspect("equal", "box")

ax.set_xlabel("Re")
ax.set_xlim(1.2, -1.2)
ax.set_xticks([1, 0, -1])

ax.set_ylabel("t")
ax.set_ylim(0, 2 * K * np.pi)
ax.set_yticks(np.linspace(0, 2 * K * np.pi, 7))
ax.set_yticklabels([])

ax.set_zlabel("Im")
ax.set_zlim(-1.2, 1.2)
ax.set_zticks([-1, 0, 1])

ax.view_init(elev=20.0, azim=40, roll=0)

step_size = 1
num_frames = t.shape[0] // step_size


def update_animation(frame):
    time = frame * step_size
    exp_plot.set_data_3d(real_vals[:time], t[:time], imag_vals[:time])
    real_plot.set_data_3d(real_vals[:time], t[:time], real_zs[:time])
    imag_plot.set_data_3d(imag_zs[:time], t[:time], imag_vals[:time])
    return [exp_plot, real_plot, imag_plot]


fig = plt.gcf()
animation = FuncAnimation(
    fig=fig, func=update_animation, frames=num_frames, interval=25
)
#
plt.show()
