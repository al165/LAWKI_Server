import os

from pythonosc.udp_client import SimpleUDPClient

from lawki_constants import *


client = SimpleUDPClient('127.0.0.1', 1881)

print('LAWKI NOW sensor emulator')
print('sensor messages: <tile|a> <0|1>')
print('"q" to quit, "a" all stations')

while True:
    try:
        msg = input('> ')
    except KeyboardInterrupt:
        print()
        break

    try:
        if msg[0] == 'q':
            break
        sn, val = msg.split()
    except:
        print('could not parse', msg)
        continue

    if sn == "a":
        for i in range(len(STATION_VIDEO_CONFIG[1]["tile_params"])):
            client.send_message('/sensor', (i, int(val)))
            print(f'sent {val} to tile {i}')
    else:
        client.send_message('/sensor', (int(sn), int(val)))
