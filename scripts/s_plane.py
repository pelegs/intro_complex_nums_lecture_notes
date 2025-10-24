import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.backend_bases import MouseButton

num_digits = 4
num_steps = 1000
t_max = 10 * np.pi
ts = np.linspace(0, t_max, num_steps)
z = 0.0 + 1.0j
ws = np.exp(z * ts)
round_threshold = 0.025

# colors?
# later


def plot_z():
    global z, cmplx_func, real_func, imag_func
    ws = np.exp(z * ts)
    cmplx_func.remove()
    (cmplx_func,) = ax_complex.plot(np.real(ws), np.imag(ws), linewidth=2, color="red")
    real_func.remove()
    (real_func,) = ax_real.plot(ts, np.real(ws), linewidth=2, color="blue")
    imag_func.remove()
    (imag_func,) = ax_imag.plot(ts, np.imag(ws), linewidth=2, color="green")
    plt.gcf().canvas.draw_idle()


def on_scroll(event):
    global t_max, ts, ws
    t_max += event.step * np.pi
    if t_max <= 0:
        t_max = 0
    ts = np.linspace(0, t_max, num_steps)
    ws = np.exp(z * ts)
    plot_z()


def on_move(event):
    global z
    if event.inaxes:
        on_move
        a = round(event.xdata, num_digits)
        b = round(event.ydata, num_digits)
        z = a + b * 1.0j
        op = "+" if b >= 0 else "-"
        z_coord_text.set_text(f"z={a:0.2f} {op} {abs(b):0.2f} i")
        plot_z()


# -------- #
#  Figures #
# -------- #

fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
fig.canvas.manager.set_window_title("The complex S-plane")

ax_s_plane = axes[0, 0]
ax_complex = axes[0, 1]
ax_real = axes[1, 0]
ax_imag = axes[1, 1]

ax_s_plane.grid()
ax_s_plane.set_aspect("equal", "box")
ax_s_plane.set_xlabel("Real")
ax_s_plane.set_ylabel("Imaginary")
ax_s_plane.set_xlim(-1.2, 1.2)
ax_s_plane.set_ylim(-1.2, 1.2)
# z_coords_annotation = ax_s_plane.annotate(
#     f"z={np.real(z)} + {np.imag(z)} i", (0, 0.5), fontsize=20
# )
z_coord_text = ax_s_plane.text(
    1.15,
    1.15,
    "z = 0.0 + 1.0 i",
    ha="right",
    va="top",
    size=20,
)
ax_s_plane.axvline(0, color="black")
ax_s_plane.axhline(0, color="black")

ax_complex.grid()
ax_complex.set_aspect("equal", "box")
ax_complex.set_xlabel("Real")
ax_complex.set_ylabel("Imaginary")
ax_complex.set_xlim(-2, 2)
ax_complex.set_ylim(-2, 2)
(cmplx_func,) = ax_complex.plot(np.real(ws), np.imag(ws), linewidth=2)

ax_real.grid()
ax_real.set_aspect("equal", "box")
ax_real.set_xlabel("t")
ax_real.set_ylabel("Imaginary")
ax_real.set_xlim(0, ts[-1])
ax_real.set_ylim(-ts[-1] / 2, ts[-1] / 2)
(real_func,) = ax_real.plot(ts, np.real(ws), linewidth=2, color="blue")

ax_imag.grid()
ax_imag.set_aspect("equal", "box")
ax_imag.set_xlabel("t")
ax_imag.set_ylabel("Imaginary")
ax_imag.set_xlim(0, ts[-1])
ax_imag.set_ylim(-ts[-1] / 2, ts[-1] / 2)
(imag_func,) = ax_imag.plot(ts, np.real(ws), linewidth=2, color="green")

plot_z()


# ------------------ #
#  Connecting stuff  #
# ------------------ #

plt.connect("motion_notify_event", on_move)
plt.connect("scroll_event", on_scroll)
plt.show()
