from bleak import *

from callbacks.motion import *
from utils.UUIDs import  *
import struct
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

async def scan():
    """
    Scan for BLE devices
    :return: A list of BLE devices found
    """
    print(f"{datetime.datetime.now()} | Scanning...")
    devices = await BleakScanner.discover(timeout=10)

    print(f"{datetime.datetime.now()} | End scan...")

    return devices

#Find my device with OpenEarable's address between all BLE devices
def find(discovered_devices, addresses):
    my_devices = []
    for i in range(len(discovered_devices)):
        if discovered_devices[i].address in addresses:
            my_devices.append(discovered_devices[i])
            # Avoid useless checks
            if len(my_devices) == len(addresses):
                break

    return my_devices

#Connection my device
async def connection(devices):
    clients = await asyncio.gather(*[connect(device) for device in devices])

    return [client for client in clients if client is not None]

async def connect(device):
    """
    Connect to a BLE device
    :param device:
    :return: BleakClient instance if connected, None otherwise
    """
    client = BleakClient(get_uuid(device))
    await client.connect()

    if client.is_connected:
        print(f"Connected to {device.address}")
        battery_level = await client.read_gatt_char(BATTERY_LEVEL_CHARACTERISTIC)
        print(f'Battery Level: {int(battery_level[0])}%')
        # Perform operations with the client here
        await change_status(client, "connected")
        return client
    else:
        print(f"Failed to connect to {device.address}")
        return None


def get_uuid(device):
    """
    Get the MAC address of a device, useful in macOS
    :param device: BLE Device
    :return: MAC address
    """
    return str(device.address)

#Change status connected/recording and write this state with RGB Led
async def change_status(client, status: str):
    payload = None

    if status == "connected":
        format_str = "<4B"
        green = (0, 255, 0)
        constant_light = 1
        payload = struct.pack(format_str, *green, constant_light)

    if status == "recording":
        format_str = "<4B"
        red = (255, 0, 0)
        constant_light = 1
        payload = struct.pack(format_str, *red, constant_light)

    if payload is not None:
        await client.write_gatt_char(LED_STATE_CHARACTERISTIC, payload)


def create_payload(sensor_id, sample_rate, latency):
    return struct.pack('<BfI', *[sensor_id, sample_rate, latency])

sensor_id = [0, 1]
payload_imu_start = create_payload(sensor_id[0], sample_rate=30.0, latency=0)
payload_bme_start = create_payload(sensor_id[1], sample_rate=30.0, latency=0)

payload_imu_stop = create_payload(sensor_id[0], sample_rate=30.0, latency=0)
payload_bme_stop = create_payload(sensor_id[1], sample_rate=30.0, latency=0)

async def receive_data_from_client(client):

    tasks = []
    tasks.append(client.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_imu_start))
    #tasks.append(client.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_bme_start))
    tasks.append(client.start_notify(SENSOR_DATA_CHARACTERISTIC, imu_data_callback))
    #tasks.append(client.start_notify(SENSOR_DATA_CHARACTERISTIC, bme_data_callback))

    await asyncio.gather(*tasks)
    await change_status(client, "recording")


# Funzione per ricezione dati illimitata
async def receive_data(clients):
    try:
        tasks = []
        for client in clients:
            tasks.append(asyncio.ensure_future(receive_data_from_client(client)))

        await asyncio.gather(*tasks)  # Aspetta che tutte le attività siano eseguite

        while True:
            await asyncio.sleep(0.1)

    except asyncio.exceptions.CancelledError or KeyboardInterrupt as _:
        print("\nStopping...")
        for client in clients:
            print(f"\nDisconnecting...")
            await client.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_imu_stop)
            #await client.write_gatt_char(SENSOR_CONFIGURATION_CHARACTERISTIC, payload_bme_stop)
            await client.disconnect()