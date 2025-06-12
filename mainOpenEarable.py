import sys
sys.coinit_flags = 0

try:
    from bleak.backends.winrt.util import allow_sta

    allow_sta()
except ImportError:
    pass

import asyncio
from utils.utility import scan, find
from classes import OpenEarableClient


async def main():
    my_openearable_addresses = ["54:EE:39:71:B5:21"]

    discovered_devices = await scan()
    my_devices = find(discovered_devices, my_openearable_addresses)

    openerable = OpenEarableClient.OpenEarableClient(my_devices[0])
    await openerable.connect()
    input("Press enter to record data... ")
    await openerable.receive_inertial_data()

if __name__ == '__main__':
    asyncio.run(main())