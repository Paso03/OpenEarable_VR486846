import os
import datetime
import struct
import numpy as np
import math


def compute_rpy(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z):
    # Calcolo Roll e Pitch
    roll = math.atan2(acc_y, acc_z)
    pitch = math.atan2(-acc_x, math.sqrt(acc_y**2 + acc_z**2))

    # Correzione del magnetometro per Roll e Pitch
    mag_x_corr = mag_x * math.cos(pitch) + mag_z * math.sin(pitch)
    mag_y_corr = mag_x * math.sin(roll) * math.sin(pitch) + mag_y * math.cos(roll) - mag_z * math.sin(roll) * math.cos(pitch)

    # Calcolo dello Yaw
    yaw = math.atan2(-mag_y_corr, mag_x_corr)

    # Conversione in gradi
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    return roll_deg, pitch_deg, yaw_deg


def imu_data_callback(sender, data):
    global roll, pitch, yaw
    folder_name = "training/data"
    os.makedirs(folder_name, exist_ok=True)
    receive_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        sensor_id = struct.unpack('<B', data[0:1])
        size = struct.unpack('<B', data[1:2])
        timestamp = struct.unpack('<I', data[2:6])

        if sensor_id[0] == 0:
            acc_x = struct.unpack('<f', data[6:10])[0]
            acc_y = struct.unpack('<f', data[10:14])[0]
            acc_z = struct.unpack('<f', data[14:18])[0]

            gyro_x = struct.unpack('<f', data[18:22])[0]
            gyro_y = struct.unpack('<f', data[22:26])[0]
            gyro_z = struct.unpack('<f', data[26:30])[0]

            mag_x = struct.unpack('<f', data[30:34])[0]
            mag_y = struct.unpack('<f', data[34:38])[0]
            mag_z = struct.unpack('<f', data[38:42])[0]

            # Calcolo RPY
            roll_deg, pitch_deg, yaw_deg = compute_rpy(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z)

            file_path = os.path.join(folder_name, ".csv")
            with open(file_path, "a+") as file:
                file.write(f"{receive_time},{acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z},{mag_x},{mag_y},{mag_z},{roll_deg},{pitch_deg},{yaw_deg}\n")
            print(f"\r{receive_time} - Acc: X={acc_x: 2.3f}, Y={acc_y: 2.3f}, Z={acc_z: 2.3f} | RPY: Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°", end="", flush=True)

    except struct.error:
        print(f"\nErrore nel parsing dei dati di: {data}")

def bme_data_callback(sender, data):
    folder_name = "training/data"
    os.makedirs(folder_name, exist_ok=True)
    receive_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    try:
        sensor_id = struct.unpack('<B', data[0:1])
        size = struct.unpack('<B', data[1:2])
        timestamp = struct.unpack('<I', data[2:6])

        if sensor_id[0] == 1:
            temp = struct.unpack('<f', data[6:10])[0]
            pres = struct.unpack('<f', data[10:14])[0]

            file_path = os.path.join(folder_name, "BME_sensor_data.csv")
            with open(file_path, "a+") as file:
                file.write(f"{receive_time},{temp},{pres}\n")
            print(f"\r{receive_time} - Temp={temp: 2.3f}°C, Pres={pres: 2.3f}Pa", end="", flush=True)

    except struct.error:
        print(f"\nErrore nel parsing dei dati di: {data}")
