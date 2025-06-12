import os
import math
import struct
from datetime import datetime
from bleak import BleakClient, BLEDevice
import onnxruntime as ort

from utils.UUIDs import *
from utils.utility import *

class OpenEarableClient(BleakClient):

    def __init__(self, device: BLEDevice):
        super().__init__(get_uuid(device))
        self.path = "training/data"
        self.mac_address = device.address

        self.model = ort.InferenceSession('training/CNN_30.onnx')
        self.classes = ["Testa_ferma", "Inclinazione_testa_dx-sx", "Nodding_testa", "Rotazione_testa_dx-sx"]

        self.buffer_size = 30
        self.data_buffer = []

        self.files = {}  # {'IMU': file_object, 'BME': file_object}

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def open_file(self, sensor_type: str):
        filename = f"{self.path}/{sensor_type}.csv"
        file = open(filename, "a+")
        self.files[sensor_type] = file

    def close_files(self):
        for f in self.files.values():
            f.close()
        self.files.clear()

    async def connect(self, **kwargs) -> bool:
        try:
            await super().connect(**kwargs)
            print(f"Successfully connected to {self.mac_address}")
            await change_status(self, "connected")
            battery_level = await self.read_gatt_char(BATTERY_LEVEL_CHARACTERISTIC)
            print(f'Battery Level: {int(battery_level[0])}%')
            return True

        except Exception as e:
            print(f"Failed to connect to {self.address}: {e}")
            return False

    def disconnect(self) -> bool:
        print(f"\nDisconnecting from {self.mac_address}")
        self.close_files()
        return super().disconnect()

    async def receive_inertial_data(self):
        tasks = []
        tasks.append(self.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_imu_start))
        #tasks.append(self.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_bme_start))
        tasks.append(self.start_notify(SENSOR_DATA_CHARACTERISTIC, self.imu_data_callback))
        #tasks.append(self.start_notify(SENSOR_DATA_CHARACTERISTIC, self.bme_data_callback))

        await asyncio.gather(*tasks)
        await change_status(self, "recording")

        # Apri i file di output
        self.open_file("IMU")
        #self.open_file("BME")  # Sblocca se usi anche i dati ambientali

        try:
            while True:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            print("\nStopped notification")
            await self.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_imu_stop)
            #await self.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_bme_stop)
            await self.disconnect()

    def compute_rpy(self, acc_x, acc_y, acc_z, mag_x, mag_y, mag_z):
        roll = math.atan2(acc_y, acc_z)
        pitch = math.atan2(-acc_x, math.sqrt(acc_y ** 2 + acc_z ** 2))

        mag_x_corr = mag_x * math.cos(pitch) + mag_z * math.sin(pitch)
        mag_y_corr = (mag_x * math.sin(roll) * math.sin(pitch) + mag_y * math.cos(roll)
                      - mag_z * math.sin(roll) * math.cos(pitch))

        yaw = math.atan2(-mag_y_corr, mag_x_corr)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def imu_data_callback(self, sender, data):
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

                roll_deg, pitch_deg, yaw_deg = self.compute_rpy(acc_x, acc_y, acc_z, mag_x, mag_y, mag_z)

                if "IMU" in self.files:
                    self.files["IMU"].write(f"{receive_time},{acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z},"
                                            f"{mag_x},{mag_y},{mag_z},{roll_deg},{pitch_deg},{yaw_deg}\n")

                if len(self.data_buffer) == self.buffer_size:
                    input_data = np.array(self.data_buffer, dtype=np.float32).reshape(1, self.buffer_size, 6)
                    input_name = self.model.get_inputs()[0].name
                    cls_index = np.argmax(self.model.run(None, {input_name: input_data})[0], axis=1)[0]
                    print(f"\r{self.mac_address} | {receive_time} - "
                          f"Prediction: {self.classes[cls_index]}", end="", flush=True)
                    self.data_buffer.clear()

                self.data_buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

                #print(f"\r{self.mac_address} | {receive_time} - Accelerometer: X={acc_x: 2.3f}, Y={acc_y: 2.3f}, Z={acc_z: 2.3f} "
                #      f"| RPY: Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°, Yaw={yaw_deg:.2f}°", end="", flush=True)
        except struct.error:
            print(f"\nErrore nel parsing dei dati: {data}")

    def bme_data_callback(self, sender, data):
        receive_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        try:
            sensor_id = struct.unpack('<B', data[0:1])
            size = struct.unpack('<B', data[1:2])
            timestamp = struct.unpack('<I', data[2:6])

            if sensor_id[0] == 1:
                temp = struct.unpack('<f', data[6:10])[0]
                pres = struct.unpack('<f', data[10:14])[0]

                if "BME" in self.files:
                    self.files["BME"].write(f"{receive_time},{temp},{pres}\n")

                print(f"\r{self.mac_address} | {receive_time} - Temp={temp: 2.3f}°C, Pres={pres: 2.3f}Pa", end="", flush=True)

        except struct.error:
            print(f"\nErrore nel parsing dei dati: {data}")