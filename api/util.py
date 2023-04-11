import random
from dotenv import load_dotenv
import os
import time
load_dotenv()

infura_keys = os.environ['infura_keys']
infura_keys = [i.strip() for i in infura_keys.split(',')]
def getRandomInfuraKey() -> str:

    index = random.randint(1,len(infura_keys)) - 1
    return infura_keys[index]


def numToBytes(number : int) -> str:
    # Convert the number to a bytes object of length 32
    padded_bytes = number.to_bytes(32, byteorder='big', signed=False)

    # Convert the bytes object to a hexadecimal string
    padded_hex = padded_bytes.hex()

    return padded_hex

def hexToBytes(hexString : str) -> str:

    return hexString.rjust(64, '0')

def getSecondUnix() -> int:
    return int(time.time())
