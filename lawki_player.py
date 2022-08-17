#!/usr/bin/python

import os
import sys
import time
import datetime
import argparse
import threading

import numpy as np

import pygame as pg
from pygame._sdl2.video import Window

from OpenGL.GL import *
from OpenGL.GLU import *

from pythonosc.dispatcher import Dispatcher
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import AsyncIOOSCUDPServer, BlockingOSCUDPServer

#from map_sinks import *
from logprint import logprint
from lawki_constants import *
from video_quad import VideoQuad


FPS = 25

SENSOR_MAP = {
    0: [],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5, 6]
}


SOUND_PORT = 1999


def main():
    with open('lawki_player.log', 'w') as f:
        t = time.localtime()
        f.write(f'LAWKI PLAYER\nmain()\n{time.strftime("%H:%M:%S", t)}\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1, help='station number')
    parser.add_argument('-l', default=False, action='store_true', help='set to local addresses')
    parser.add_argument('-width', type=int, default=-1, help='player width (-1 means use config)')
    parser.add_argument('-height', type=int, default=-1, help='player height (-1 means use config)')
    parser.add_argument('-v', type=str, default='', help='video path')

    args = parser.parse_args()

    try:
        conf = STATION_VIDEO_CONFIG[args.n]
    except KeyError:
        print(f'station video config not found for station {args.n}')
        conf = STATION_VIDEO_CONFIG[1]

    if args.width > 0:
        conf['width'] = args.width
    if args.height > 0:
        conf['height'] = args.height


    pg.init()
    pg.display.set_mode((conf['width'], conf['height']), pg.OPENGL | pg.DOUBLEBUF, vsync=1)
    pg.display.set_caption('LAWKI NOW')
    #pg.mixer.init()
    
    logprint(pg.mixer.get_num_channels())

    #mapping = getAudioMapping()
    #devicename = getAudioDevice(args.n, mapping)
    #if devicename:
    #    logprint(f'Switching mixer for station {args.n} to audio device {devicename}')
    #    pg.mixer.quit()
    #    pg.mixer.pre_init(devicename=devicename)
    #    pg.mixer.init()

    app = LawkiPlayer(args.n, args.l, save_log=True)

    for x, y, w, h in conf['tile_params']:
        app.add_video_tile(args.v, posX=x, posY=y, width=w, height=h)

    app.main_loop()

    print('\nEXITING\n')
    pg.quit()
    sys.exit()



class LawkiPlayer:

    def __init__(self, station_number=1, local=False, conf=None, save_log=False):
        if save_log:
            with open('lawki_player.log', 'a') as f:
                f.write('__init__')

        #os.nice(-5)

        if conf is None:
            try:
                conf = {**STATION_VIDEO_CONFIG[station_number], **STATION_ADDRESSES[station_number]}
            except KeyError:
                conf = STATION_ADDRESSES[station_number]

        if local:
            conf['addr'] = '127.0.0.1'

        self.screen = pg.display.get_surface()
        self.screen_rect = self.screen.get_rect()
        self.clock = pg.time.Clock()
        self.done = False

        window = Window.from_display_module()
        window.position = (conf['x'], conf['y'])

        self.width = self.screen_rect.width
        self.height = self.screen_rect.height

        self.shader_uniforms = FILTER_DEFAULT_UNIFORMS

        self.video_tiles = dict()
        self.current_tile = None
        #self.vid_audio = None

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.msg_recieve)

        logprint("listening on: ", conf['addr'], conf['port'])
        self.server = BlockingOSCUDPServer((conf['addr'], conf['port']), dispatcher)
        self.server.timeout = 0.001

        self.audience_prescence = 1.0

        try:
            self.client = SimpleUDPClient("127.0.0.1", SOUND_PORT)
        except Exception as e:
            logprint(e)
            logprint(f'could not create UDP Client on {SOUND_PORT}')
            self.client = None

        print('\nLAWKI NOW Player Initialised\n')
        print(f' - station number: {station_number}')
        print(f' - window position: {conf["x"]}, {conf["y"]}')
        print(f' - window size: {conf["width"]}x{conf["height"]}')
        print(f' - listening on port {conf["port"]}\n')
        sys.stdout.flush()

        if save_log:
            with open('lawki_player.log', 'a') as f:
                f.write('Done init')
                


    def add_video_tile(self, fp=None, posX=0.0, posY=0.0, width=None, height=None):
        tile_num = len(self.video_tiles) + 1
        self.video_tiles[tile_num] = VideoQuad(None, self.width, self.height, posX, posY, width, height)
        self.current_tile = tile_num

        if fp:
            msg_dst = tile_num
            self.set_video(msg_dst, fp)

        print(f'added tile {tile_num}')

        self.update_shader_uniforms()


    def msg_recieve(self, addr, *data):
        '''
        `addr` is expected to be '/station_{sn}/{type}/{tile_number}/', where `type` is one
        of ['audio', 'video', 'sensor', 'close'].

        Parameters
        ----------
        addr : str
            Address to send `data` to.
        *data : list
            Collects data args as a list.
        '''

        dest = addr.split('/')[1:]
        if len(dest) < 2:
            logprint(f' - Unkown message received: {addr}, {data}')
            return

        msg_type = dest[1]

        #logprint(data)

        try:
            msg_dst = int(dest[2])
        except (ValueError, IndexError):
            msg_dst = None

        if msg_type == 'video':
            logprint('video', msg_dst, *data)
            self.set_video(msg_dst, *data)

        elif msg_type == 'audio':
            self.play_audio_sample(msg_dst, *data)

        elif msg_type == 'sensor':
            logprint('sensor', msg_dst, *data)
            # msg_dst is the sensor number!
            self.sensor_update(msg_dst, *data)

            # relay sensor data to sound system
            if self.client:
                self.client.send_message(f'/sensor/{msg_dst}', int(data[0]))

        elif msg_type == 'close':
            self.done = True
            self.close()

        else:
            logprint(f' - Unknown message type received: {msg_type}')


    def set_video(self, msg_dst, *data):
        ''' data = [fn, log_views, start_time, audio] '''

        fn, log_views, start_time, audio = data

        #logprint(msg_dst, data)
        if len(self.video_tiles) == 0:
            return

        fp = os.path.join(VIDEO_BASE_FP, fn)
        #msg_dst = msg_dst % len(self.video_tiles)

        if fp is None or fp == '' or fp == 'BLANK':
            logprint('video fp is invalid', fp)
            #if self.vid_audio:
            #    self.vid_audio.stop()
            return

        self.video_tiles[msg_dst].set_video(fp, start_frame=int(start_time * FPS))
        self.video_tiles[msg_dst].shader_uniforms['distance'] = (log_views / 18) * 0.02

        # `audio` is the last element of `data`
        if audio:
            # get audio stream...
            # get start time in format
            ss = str(datetime.timedelta(seconds=start_time))
            
            command = f"ffmpeg -y -hide_banner -loglevel error -i '{fp}' -ss {ss} -ab 160k -ac 2 -ar 44100 -vn {os.path.join(AUDIO_TEMP_DIR, 'audio.wav')} &"
            #logprint(command)
            os.system(command)

            # save the video name in a text file
            with open('/tmp/curr_video_title.txt', 'w') as f:
                f.write(fp)

            # let sound system know to play video
            if self.client:
                self.client.send_message('/audio', 1)

            #if self.vid_audio:
            #    self.vid_audio.stop()

            #try:
            #    self.vid_audio = pg.mixer.Sound(os.path.join(AUDIO_TEMP_DIR, 'audio.wav'))
            #    self.vid_audio.play()
            #except FileNotFoundError:
            #    pass


    def play_audio_sample(msg_dst, *data):
        fp = data[0]
        if fp == 'BLANK' or fp is None:
            return

        sample = pg.mixer.Sound(fp)

        if 'vocals' not in fp and 'bass' not in fp:
            sample.set_volume(0.8)

        sample.play()

    def sensor_update(self, sensor_num, *data):

        val = int(data[0])

        if val == 1:
            decay = 0.75
        else:
            decay = 1.0

        for dst in SENSOR_MAP[sensor_num]:
            self.video_tiles[dst].shader_uniforms['decay'] = decay
            #logprint(f'setting tile {dst} decay to {decay}')


    def update_shader_uniforms(self, panel=None, clamp=True, **kwargs):
        '''shader uniform values always clamped between 0 and 1, and applied to all video tiles'''

        for k, v in kwargs.items():
            if clamp:
                self.shader_uniforms[k] = min(1.0, max(0.0, v))
            else:
                self.shader_uniforms[k] = v

        if panel is None:
            for tile in self.video_tiles.values():
                tile.shader_uniforms = dict(self.shader_uniforms)
        else:
            for k, v in kwargs.values():
                self.video_tiles[panel].shader_uniforms[k] = self.shader_uniforms[k]

        #self.print_uniforms()


    def main_loop(self):
        self.start_time = time.time()
        while not self.done:
            self.event_loop()
            self.update()
            self.render()
            self.clock.tick(FPS)
            #sys.stdout.write(f'FPS: {self.clock.get_fps():.2f}   \r')



    def update(self):
        self.server.handle_request()


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SLASH:
                    self.current_tile += 1
                    self.current_tile %= len(self.video_tiles)
                    print(f'Selected tile {self.current_tile}')
                if event.key == pg.K_z:
                    self.update_shader_uniforms(green=not self.shader_uniforms['green'])
                if event.key == pg.K_q:
                    self.update_shader_uniforms(threshold=self.shader_uniforms['threshold'] + 0.02)
                if event.key == pg.K_a:
                    self.update_shader_uniforms(threshold=self.shader_uniforms['threshold'] - 0.02)
                if event.key == pg.K_w:
                    self.update_shader_uniforms(decay=self.shader_uniforms['decay'] + 0.02)
                if event.key == pg.K_s:
                    self.update_shader_uniforms(decay=self.shader_uniforms['decay'] - 0.02)
                if event.key == pg.K_e:
                    self.update_shader_uniforms(seed=self.shader_uniforms['seed'] + 0.02)
                if event.key == pg.K_d:
                    self.update_shader_uniforms(seed=self.shader_uniforms['seed'] - 0.02)
                if event.key == pg.K_r:
                    self.update_shader_uniforms(shade=self.shader_uniforms['shade'] + 0.02)
                if event.key == pg.K_f:
                    self.update_shader_uniforms(shade=self.shader_uniforms['shade'] - 0.02)
                if event.key == pg.K_t:
                    self.update_shader_uniforms(reaction=self.shader_uniforms['reaction'] + 0.02)
                if event.key == pg.K_g:
                    self.update_shader_uniforms(reaction=self.shader_uniforms['reaction'] - 0.02)
                if event.key == pg.K_x:
                    self.update_shader_uniforms(highlight=(self.shader_uniforms['highlight'] + 1) % 5, clamp=False)

        pressed_keys = pg.key.get_pressed()

        if pressed_keys[pg.K_ESCAPE]:
            self.done = True

        # Video placement:
        if pressed_keys[pg.K_UP] and self.current_tile is not None:
            self.video_tiles[self.current_tile].posY += 1
            self.print_tile_props()
        if pressed_keys[pg.K_DOWN] and self.current_tile is not None:
            self.video_tiles[self.current_tile].posY -= 1
            self.print_tile_props()
        if pressed_keys[pg.K_LEFT] and self.current_tile is not None:
            self.video_tiles[self.current_tile].posX -= 1
            self.print_tile_props()
        if pressed_keys[pg.K_RIGHT] and self.current_tile is not None:
            self.video_tiles[self.current_tile].posX += 1
            self.print_tile_props()
        if pressed_keys[pg.K_PERIOD] and self.current_tile is not None:
            self.video_tiles[self.current_tile].scale += 1
            self.print_tile_props()
        if pressed_keys[pg.K_COMMA] and self.current_tile is not None:
            self.video_tiles[self.current_tile].scale -= 1
            self.print_tile_props()


    def render(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(0.5, 0.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        i_time = time.time() - self.start_time

        for v in self.video_tiles.values():
            v.draw(i_time)

        pg.display.flip()


    def on_resize(self, width, height):
        for v in self.video_tiles.values():
            v.window_width = width
            v.window_height = height


    def print_uniforms(self):
        fmt_string = ['{}: {:.3f}'.format(k, v) for k, v in self.shader_uniforms.items()]
        sys.stdout.write(' '.join(fmt_string) + '\r')


    def print_tile_props(self):
        tile = self.video_tiles[self.current_tile]
        sys.stdout.write(f'{tile.posX:.3f}, {tile.posY:.3f}, {tile.scale:.3f}          \n')



if __name__ == '__main__':
    main()

