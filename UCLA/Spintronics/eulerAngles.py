import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# Rotation utilities
# -------------------------
def Rz(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rotate_vector(v, α, β, γ):
    # Z-Y-Z Euler angles (change if you want a different convention)
    return Rz(α) @ Ry(β) @ Rz(γ) @ v

# -------------------------
# Initial vector and Euler angles
# -------------------------
v0 = np.array([1.0, 0.0, 0.0])
α, β, γ = 0.0, 0.0, 0.0

# points stored for path tracing
trace = []

# -------------------------
# Figure setup
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# unit sphere
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(x, y, z, color='gray', alpha=0.2, linewidth=0)

# axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Vector Precession on Sphere (press keys)")
ax.view_init(30, 40)

# Red vector
(vector_line,) = ax.plot([0, 0], [0, 0], [0, 1], lw=3, color='red')

# Blue trace curve
(trace_line,) = ax.plot([], [], [], lw=2, color='blue')

# -------------------------
# Keyboard control
# -------------------------
def on_key(event):
    global α, β, γ
    step = np.deg2rad(5)

    if event.key == "[": α += step
    if event.key == "]": α -= step
    if event.key == "[": β -= step
    if event.key == "]": β += step
    if event.key == ",": γ += step
    if event.key == ".": γ -= step

    print(f"α={np.rad2deg(α):.1f}°, β={np.rad2deg(β):.1f}°, γ={np.rad2deg(γ):.1f}°")

fig.canvas.mpl_connect("key_press_event", on_key)

# -------------------------
# Animation update
# -------------------------
def update(frame):
    global vector_line, trace_line, trace

    # update vector
    v = rotate_vector(v0, α, β, γ)

    # update red vector line
    vector_line.set_data([0, v[0]], [0, v[1]])
    vector_line.set_3d_properties([0, v[2]])

    # append to trace
    trace.append(v.copy())

    # update blue trace curve
    trace_arr = np.array(trace)
    trace_line.set_data(trace_arr[:, 0], trace_arr[:, 1])
    trace_line.set_3d_properties(trace_arr[:, 2])

    return vector_line, trace_line

ani = FuncAnimation(fig, update, interval=40, blit=True)

plt.show()
