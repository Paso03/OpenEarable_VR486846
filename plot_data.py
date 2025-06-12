import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

data_directory = "data"
recording_name_imu = os.path.join(data_directory, "IMU.csv")
recording_name_bme = os.path.join(data_directory, "BME.csv")

skip_rows_imu = 0
skip_rows_bme = 0

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Assegna i subplot correttamente
accelerometer_fig = axes[0, 0]
gyroscope_fig = axes[0, 1]
magnetometer_fig = axes[0, 2]
angle_fig = axes[1, 0]
temperature_fig = axes[1, 1]
pressure_fig = axes[1, 2]

def animate(i):
    global skip_rows_imu, skip_rows_bme, recording_name_imu, recording_name_bme

    if not os.path.exists(recording_name_imu):
        print(f"File non trovato {recording_name_imu}")
        return

    df_imu = pd.read_csv(
        recording_name_imu,
        header=None,
        names=["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z", "roll_deg", "pitch_deg", "yaw_deg"],
        index_col=0,
        parse_dates=True,
        skiprows=skip_rows_imu
    )

    if len(df_imu) >= 600:
        skip_rows_imu += len(df_imu) - 600

    accelerometer_fig.clear()
    gyroscope_fig.clear()
    magnetometer_fig.clear()
    angle_fig.clear()

    accelerometer_fig.plot(df_imu["acc_x"], color="red", label="X")
    accelerometer_fig.plot(df_imu["acc_y"], color="green", label="Y")
    accelerometer_fig.plot(df_imu["acc_z"], color="blue", label="Z")
    accelerometer_fig.set_ylabel("Magnitude")
    accelerometer_fig.set_xlabel("Time")
    accelerometer_fig.set_title("Accelerometer Data")
    accelerometer_fig.legend()

    gyroscope_fig.plot(df_imu["gyro_x"], color="yellow", label="X")
    gyroscope_fig.plot(df_imu["gyro_y"], color="magenta", label="Y")
    gyroscope_fig.plot(df_imu["gyro_z"], color="cyan", label="Z")
    gyroscope_fig.set_ylabel("Magnitude")
    gyroscope_fig.set_xlabel("Time")
    gyroscope_fig.set_title("Gyroscope Data")
    gyroscope_fig.legend()

    magnetometer_fig.plot(df_imu["mag_x"], color="fuchsia", label="X")
    magnetometer_fig.plot(df_imu["mag_y"], color="lightgreen", label="Y")
    magnetometer_fig.plot(df_imu["mag_z"], color="aqua", label="Z")
    magnetometer_fig.set_ylabel("Magnitude")
    magnetometer_fig.set_xlabel("Time")
    magnetometer_fig.set_title("Magnetometer Data")
    magnetometer_fig.legend()

    angle_fig.plot(df_imu["roll_deg"], color="red", label="Roll")
    angle_fig.plot(df_imu["pitch_deg"], color="blue", label="Pitch")
    angle_fig.set_ylabel("Angle")
    angle_fig.set_xlabel("Time")
    angle_fig.set_title("RPY Data")
    angle_fig.legend()

    if not os.path.exists(recording_name_bme):
        print(f"File non trovato {recording_name_bme}")
        return

    df_bme = pd.read_csv(
        recording_name_bme,
        header=None,
        names=["time", "temp", "pres"],
        index_col=0,
        parse_dates=True,
        skiprows=skip_rows_bme
    )

    if len(df_bme) >= 600:
        skip_rows_bme += len(df_bme) - 600

    temperature_fig.clear()
    pressure_fig.clear()

    temperature_fig.plot(df_bme["temp"], color="darksalmon", label="Temp")
    temperature_fig.set_ylabel("°C")
    temperature_fig.set_xlabel("Time")
    temperature_fig.set_title("Temperature Data")
    temperature_fig.legend()

    pressure_fig.plot(df_bme["pres"], color="limegreen", label="Pres")
    pressure_fig.set_ylabel("Pa")
    pressure_fig.set_xlabel("Time")
    pressure_fig.set_title("Pressure Data")
    pressure_fig.legend()

    fig.suptitle("Wearable Data", fontsize=14, fontweight="bold")
    plt.tight_layout()

def live_plotting():
    ani = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
    plt.show()

def main():
    live_plotting()

if __name__ == '__main__':
    main()
