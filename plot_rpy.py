import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import datetime

# === Configurazione file ===
recording_name_rpy = 'training/data/IMU.csv'         # file CSV contenente roll, pitch, yaw
model_file = 'mesh/OpenEarable.obj'                  # file .obj del modello 3D

# === Funzione per leggere file OBJ ===
def load_obj(filename):
    vertices, faces = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.strip().split()[1:]])
            elif line.startswith('f '):
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), np.array(faces)

# === Rotazione e scala ===
def rotate_model(vertices, roll, pitch, yaw, scale_factor=1.0):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rotated = (R @ (vertices * scale_factor).T).T
    return rotated, R

# === Caricamento modello ===
model_vertices, model_faces = load_obj(model_file)

# === Setup figura ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(top=0.85)  # spazio per il titolo

# === Legenda assi ===
legend_elements = [
    Line2D([0], [0], color='r', lw=2, label='X axis'),
    Line2D([0], [0], color='g', lw=2, label='Y axis'),
    Line2D([0], [0], color='b', lw=2, label='Z axis')
]

# === Variabili globali ===
last_timestamp = None
last_print_time = time.time()

# === Funzione animazione ===

def animate(i):
    global last_timestamp

    try:
        df = pd.read_csv(
            recording_name_rpy,
            header=None,
            names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                   "mag_x", "mag_y", "mag_z", "roll_deg", "pitch_deg", "yaw_deg"],
            usecols=["time", "roll_deg", "pitch_deg", "yaw_deg"]
        ).tail(1)
    except Exception as e:
        print("Errore lettura:", e)
        return

    # Convertiamo la colonna time in datetime
    df["time"] = pd.to_datetime(df["time"])

    last = df.iloc[-1]
    current_time = last["time"]

    # Controllo se è lo stesso timestamp del frame precedente
    if last_timestamp is not None and current_time == last_timestamp:
        return

    last_timestamp = current_time

    roll = np.radians(last["roll_deg"])
    pitch = np.radians(last["pitch_deg"])
    yaw = np.radians(last["yaw_deg"])

    rotated_vertices, R = rotate_model(model_vertices, roll, pitch, yaw, scale_factor=0.2)
    center = np.mean(rotated_vertices, axis=0)

    ax.cla()
    ax.plot_trisurf(rotated_vertices[:, 0], rotated_vertices[:, 1], rotated_vertices[:, 2],
                    triangles=model_faces, color='black', alpha=0.3)

    # Assi
    axis_length = 2
    directions = [R @ np.array([axis_length, 0, 0]),
                  R @ np.array([0, axis_length, 0]),
                  R @ np.array([0, 0, axis_length])]
    colors = ['r', 'g', 'b']
    for d, c in zip(directions, colors):
        ax.quiver(*center, *d, color=c)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.suptitle(f"3D Orientation\n"
                 f"Roll: {last['roll_deg']:.2f}°, Pitch: {last['pitch_deg']:.2f}°, Yaw: {last['yaw_deg']:.2f}°", fontsize=14)
    ax.legend(handles=legend_elements, loc='lower left')

# === Funzione principale ===
def main():
    ani = FuncAnimation(fig, animate, interval=100, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()