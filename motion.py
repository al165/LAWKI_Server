from gpiozero import MotionSensor
from time import time, sleep

from pythonosc.udp_client import SimpleUDPClient

PINS = {
    0: 17,
    1: 18,
    2: 27,
    3: 22,
    4: 23,
}


# Use broadcast address to send to all devices on the network
ADDR = "255.255.255.255"
PORT = 1881

client = SimpleUDPClient(ADDR, PORT, allow_broadcast=True)

COOLDOWN = 60

PIRs = dict([(i, MotionSensor(p, pull_up=False, queue_len=10))
             for i, p in PINS.items()])

active = dict([(i, False) for i in PINS.keys()])
cooldown_time = dict([(i, 0.0) for i in PINS.keys()])
cooldown_start = dict([(i, False) for i in PINS.keys()])

print('== starting ==')
print(f'{ADDR} {PORT}')
print()

while True:
    sleep(0.05)

    for i, pir in PIRs.items():
        if pir.value > 0.4 and not active[i]:
            print(f"[ {i} ] someone")
            active[i] = True
            cooldown_start[i] = False
            client.send_message("/sensor", (i, 1))
        else:
            if not cooldown_start[i]:
                cooldown_start[i] = True
                cooldown_time[i] = time()
            elif time() > cooldown_time[i] + COOLDOWN:
                active[i] = False
                print(f"[ {i} ] nobody")
                cooldown_start[i] = False
                client.send_message("/sensor", (i, 0))
            else:
                # waiting before setting people to False
                pass
