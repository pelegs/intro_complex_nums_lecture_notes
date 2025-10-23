import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton


def on_click(event):
    if event.button is MouseButton.LEFT:
        print(event.xdata, event.ydata)


num_steps = 250
ts = np.linspace(0, 2 * np.pi, num_steps)
z = -0.05 + 1.0j
# temp
ws = np.exp(z * ts)

# -------- #
#  Figures #
# -------- #

gs = gridspec.GridSpec(1, 2, height_ratios=[1] * 1, width_ratios=[1] * 2)

ax_s_plane = plt.subplot(gs[0, 0])
ax_func = plt.subplot(gs[0, 1])

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
ax_func.plot(np.real(ws), np.imag(ws), linewidth=2)

# ------------------ #
#  Connecting stuff  #
# ------------------ #

plt.connect("button_press_event", on_click)
plt.show()
