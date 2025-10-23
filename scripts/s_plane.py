import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backend_bases import MouseButton

num_digits = 5
num_steps = 1000
t_max = 10 * np.pi
ts = np.linspace(0, t_max, num_steps)
z = -0.2 + 1.0j
# temp
ws = np.exp(z * ts)


def plot_z(z):
    global cmplx_func
    ws = np.exp(z * ts)
    cmplx_func.remove()
    (cmplx_func,) = ax_func.plot(np.real(ws), np.imag(ws), linewidth=2, color="blue")
    plt.gcf().canvas.draw_idle()


def on_move(event):
    global z
    if event.inaxes:
        on_move
        a = round(event.xdata, num_digits)
        b = round(event.ydata, num_digits)
        z = a + b * 1.0j
        plot_z(z)
        # print(f"z_old = {z}")
        # print(f"z_new = {new_z}")


# -------- #
#  Figures #
# -------- #

fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True)

ax_s_plane = axes[0]
ax_func = axes[1]

ax_s_plane.grid()
ax_s_plane.set_aspect("equal", "box")
ax_s_plane.set_xlabel("Real")
ax_s_plane.set_ylabel("Imaginary")
ax_s_plane.set_xlim(-2, 2)
ax_s_plane.set_ylim(-2, 2)

ax_func.grid()
ax_func.set_aspect("equal", "box")
ax_func.set_xlabel("Real")
ax_func.set_ylabel("Imaginary")
ax_func.set_xlim(-2, 2)
ax_func.set_ylim(-2, 2)
(cmplx_func,) = ax_func.plot(np.real(ws), np.imag(ws), linewidth=2)
plot_z(z)

# ------------------ #
#  Connecting stuff  #
# ------------------ #

plt.connect("motion_notify_event", on_move)
plt.show()
