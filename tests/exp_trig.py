import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

ax = plt.figure().add_subplot(projection="3d")

num_ticks = 100
t = np.linspace(0, 3 * np.pi, num_ticks)
x = np.linspace(0, 2 * np.pi, num_ticks)
im_xs = np.ones(num_ticks)
real_zs = -1 * np.ones(num_ticks)

(exp_plot,) = ax.plot(
    xs=[0, 1], ys=[0, 0], zs=0, zdir="y", label="exp(it)", linewidth=2
)
(im_plot,) = ax.plot(t, np.sin(t), zs=1, zdir="x", label="sin(t)", linewidth=2)
(im_dashes,) = ax.plot([1, 1], [0, 0], zs=0, zdir="y", c="gray")
(real_plot,) = ax.plot(np.cos(t), t, zs=-1, label="cos(t)", linewidth=2)
(real_dashes,) = ax.plot([1, 1], [0, 0], zs=0, zdir="y", c="gray")


ax.set_aspect("equal", "box")

ax.set_xlabel("Re")
ax.set_xlim(1.2, -1.2)
ax.set_xticks([1, 0, -1])

ax.set_ylabel("t")
ax.set_ylim(0, 3 * np.pi)
ax.set_yticks(np.linspace(0, 3 * np.pi, 7))
ax.set_yticklabels([])

ax.set_zlabel("Im")
ax.set_zlim(-1.2, 1.2)
ax.set_zticks([-1, 0, 1])
ax.view_init(elev=20.0, azim=40, roll=0)


def update_animation(frame):
    ph = 2 * np.pi / num_frames * frame
    exp_plot.set_data_3d([0, np.cos(ph)], [0, 0], [0, np.sin(ph)])

    im_plot.set_data_3d(im_xs, t, np.sin(t + ph))
    real_plot.set_data_3d(np.cos(t + ph), t, real_zs)

    im_dashes.set_data_3d([np.cos(ph), 1], [0, 0], [np.sin(ph), np.sin(ph)])
    real_dashes.set_data_3d([np.cos(ph), np.cos(ph)], [0, 0], [np.sin(ph), -1])

    return [exp_plot, im_plot, real_plot, im_dashes, real_dashes]


num_frames = 36 * 4
fig = plt.gcf()
animation = FuncAnimation(fig=fig, func=update_animation, frames=num_frames, interval=0)

plt.show()
