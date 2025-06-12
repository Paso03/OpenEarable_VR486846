import sys
sys.coinit_flags = 0

try:
    from bleak.backends.winrt.util import allow_sta

    allow_sta()
except ImportError:
    pass

import asyncio
from utils.utility import *


async def main():
    #OpenEarable's address
    my_device = ["54:EE:39:71:B5:21"]
    devices = await scan()
    openearable = find(devices, my_device)

    # Find my device with OpenEarable's address
    client = await connection(openearable)

    input("Press enter to record data... ")

    await receive_data(client)

if __name__ == '__main__':
    asyncio.run(main())

