import os
import sys
import time
import math
import random
import pickle as pkl

import numpy as np
#import pandas as pd

#import pygame as pg

import cv2

from OpenGL.GL import *
import OpenGL.GL.shaders

from etgg2801 import Matrix4
from logprint import logprint
from shaders.shaders import *
from lawki_constants import *



#DEFAULT_CENTERS = pd.Series({'centre_0': '[20  0  20]', 'centre_1': '[200  250  220]', 'centre_2': '[240  200  12]', 'centre_3': '[50  20  200]', 'centre_4': '[110  10  150]', 'centre_5': '[80  200  230]'})


class VideoQuad:
    '''Following https://learnopengl.com/Getting-started/Textures '''

    def __init__(self, video_path, window_width, window_height, posX=0, posY=0, width=None, height=None):

        assert window_height > 0
        assert window_width > 0

        self.window_width = window_width
        self.window_height = window_height


        if width is None:
            logprint('width not specified, setting to full width')
            self.width = window_width
        else:
            self.width = width

        if height is None:
            logprint('height not specified, setting to full height')
            self.height = window_height
        else:
            self.height = height

        self.scale = self.height / 2

        # aspect of this quad (region)
        self.r_aspect = self.width / self.height

        # posX: 0 -> -scale, window_width -> scale
        self.posX = posX  #((posX / window_width) * 2) * self.scale

        # posY: 0 -> 0, window_height -> -2.0
        self.posY = posY  # -(posY / window_height) * 2


        self.tex_width = int(window_width)
        self.tex_height = int(window_height)

        logprint(f'{window_width}, {window_height}, {posX}, {posY}, {width}x{height}, {self.scale}')

        self.shader_uniforms = {
            'threshold': 1.0,
            'decay': 0.7,
            'seed': 0.5,
            'reaction': 0.0,
            'scaleY': 1.0,
            'green': 1.0,
            'highlight': 1,
            'resolution': [int(self.width), int(self.height)],
        }

        self.kmeans = None

        self.shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER_BASE, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER_BASE, GL_FRAGMENT_SHADER),
        )

        self.compute_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(VERTEX_SHADER_COMPUTE, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER_COMPUTE, GL_FRAGMENT_SHADER),
        )

        self.vertices = np.array([
            #positions        #colors          #tex coords
           -1.0, -1.0, 0.0,   1.0, 0.0, 0.0,   0.0, 0.0,      # 0 bottom left
            1.0, -1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,      # 1 bottom right
            1.0,  1.0, 0.0,   0.0, 0.0, 1.0,   1.0, 1.0,      # 2 top right
           -1.0,  1.0, 0.0,   1.0, 1.0, 1.0,   0.0, 1.0,      # 3 top left
        ], dtype=np.float32)

        # create buffers...
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # 4bytes for float32
        glBufferData(GL_ARRAY_BUFFER, len(self.vertices)*4, self.vertices, GL_STATIC_DRAW)

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # for the compute shader
        self.compute_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.compute_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.tex_width, self.tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.compute_texture, 0)

        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

        # position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        # color attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        # texture coord attribute
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

        # projection_loc
        self.projection_loc = glGetUniformLocation(self.shader, b"a_projection")
        self.model_loc = glGetUniformLocation(self.shader, b"a_model")
        self.texture1_loc = glGetUniformLocation(self.shader, b"texture1")
        self.texture2_loc = glGetUniformLocation(self.shader, b"texture2")
        self.video_src_loc = glGetUniformLocation(self.compute_shader, b"video_src")
        self.mask_loc = glGetUniformLocation(self.compute_shader, b"mask")
        self.prev_frame_loc = glGetUniformLocation(self.compute_shader, b"prev_frame")
        self.time_loc = glGetUniformLocation(self.compute_shader, b"time")
        self.center_loc = glGetUniformLocation(self.compute_shader, b"center")
        self.uniform_locations = dict([(k, glGetUniformLocation(self.compute_shader, bytes(k, 'ascii'))) for k in self.shader_uniforms.keys()])


        # create video_texture
        self.video_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.video_texture)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # initialise and start playing video source

        #try:
        #    self.cluster_df = pd.read_csv('./cluster_centres_big.csv').set_index(['vid_id', 'snip_nr'])
        #except FileNotFoundError:
        #    #print('clusters dataframe not found')
        #    self.cluster_df = None

        #try:
        #    self.snippet_df = pd.read_csv('./snippets.csv').set_index(['vid_id', 'snip_nr'])
        #except FileNotFoundError:
        #    print('snipped dataframe not found')
        #    self.snippet_df = None


        self.clusters = None
        self.video_cap = None
        self.set_video(video_path)


    def set_video(self, fp, start_frame=0, log_views=2.0, audio=True):

        def str2list(string):
            return string.strip('[]').split()

        if fp is None:
            self.video_cap = None
            return

        id = os.path.split(fp)[-1][:-4]

        if not os.path.exists(os.path.abspath(fp)):
            self.video_cap = None
            logprint(f'{fp} does not exist')
            return


        if self.video_cap:
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(os.path.abspath(fp))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            _, frame = self.video_cap.read()
        except Exception as e:
            logprint(e)
            self.video_cap.release()
            return

        if frame is None:
            logprint('frame is None')
            self.video_cap.release()
            return

        self.f_height, self.f_width, self.depth = frame.shape
        self.f_aspect = self.f_width / self.f_height

        self.center = frame[::64].mean(axis=(0,1)) / 255
        #self.center = np.array([0.5, 0.0, 1.0])


    def draw(self, i_time):
        if self.video_cap is None or not self.video_cap.isOpened():# or self.clusters is None:
            return

        glBindTexture(GL_TEXTURE_2D, self.video_texture)

        # get next frame from video stream
        ret_val, frame = self.video_cap.read()
        if not ret_val and self.video_cap.isOpened():
            # reset the frame counter
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_val, frame = self.video_cap.read()

        # convert colour and flip
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.f_aspect > self.r_aspect:
            # use frames height, calculate width
            region_height = self.f_height
            region_width = int(self.r_aspect * region_height)

            w_start_index = int((self.f_width - region_width) / 2)
            region = frame[:, w_start_index:w_start_index+region_width]

        else:
            # use frames width, calculate height
            region_width = self.f_width
            region_height = int(region_width / self.r_aspect)

            h_start_index = int((self.f_height - region_height) / 2)
            region = frame[h_start_index:h_start_index+region_height, :]


        #print(f'frame.shape {frame.shape}')

        if self.depth == 3:
            profile = GL_RGB
        elif self.depth == 4:
            profile = GL_RGBA
        else:
            print('unknown depth', depth)
            self.video_cap.release()
            self.video_cap = None
            return


        glTexImage2D(GL_TEXTURE_2D, 0, profile, region_width, region_height, 0, profile, GL_UNSIGNED_BYTE, region)

        # render the compute shader to compute_texture
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.tex_width, self.tex_height)  # the size of the compute texture
        glUseProgram(self.compute_shader)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.video_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.compute_texture)

        glUniform1i(self.video_src_loc, 0)
        glUniform1i(self.prev_frame_loc, 1)
        glUniform1f(self.time_loc, i_time)

        glUniform1f(self.uniform_locations["seed"], self.shader_uniforms['seed'])
        glUniform1f(self.uniform_locations["decay"], self.shader_uniforms['decay'])
        glUniform1f(self.uniform_locations["reaction"], self.shader_uniforms['reaction'])
        glUniform1f(self.uniform_locations["threshold"], self.shader_uniforms['threshold'])
        glUniform2f(self.uniform_locations["resolution"], self.tex_width, self.tex_height)

        #glUniform3fv(glGetUniformLocation(self.compute_shader, b"clusterCenters"), 6, self.clusters)
        #glUniform1i(glGetUniformLocation(self.compute_shader, b"highlight"), self.shader_uniforms['highlight'])

        glUniform3f(self.center_loc, *self.center) #center[0], center[1], center[2])

        glDrawArrays(GL_QUADS, 0, len(self.vertices))

        # draw quad on display buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_width, self.window_height)
        glUseProgram(self.shader)

        projection = Matrix4.getOrthographic(top=0, bottom=self.window_height,
                                             left=0, right=self.window_width,
                                             near=-1.0, far=10.0)
        #projection = Matrix4.getOrthographic(near=-1.0, far=10.0)
        #projection.set(0, 0, projection.get(0, 0) * (self.window_height / self.window_width))

        #model *= Matrix4.getScale(sx=r_aspect, sy=r_aspect, sz=1.0)                         # | fill screen
        #model = Matrix4.getScale(sx=self.scale, sy=self.scale, sz=1.0)                 # | quad scale           
        #        * Matrix4.getTranslation(self.posX, self.posY, 0.0)                       # A quad position        
        #model *= Matrix4.getScale(sx=self.width / self.window_width,
        #                          sy=self.height / self.window_height, sz=1.0)    # | quad scale width
        #        * Matrix4.getScale(sx=r_aspect)                                            # | quad aspect ratio    


        model = Matrix4.getTranslation(self.posX, self.posY, 0.0)  \
                      * Matrix4.getScale(self.scale, self.scale, 1.0) \
                      * Matrix4.getScale(sx=self.r_aspect) \
                      * Matrix4.getTranslation(1.0, 1.0, 0.0)

        glUniformMatrix4fv(self.projection_loc, 1, False, projection.getCType())
        glUniformMatrix4fv(self.model_loc, 1, False, model.getCType())

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.video_texture)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.compute_texture)

        glUniform1i(self.texture1_loc, 0)
        glUniform1i(self.texture2_loc, 1)

        glDrawArrays(GL_QUADS, 0, len(self.vertices))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)



