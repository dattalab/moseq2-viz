import os
import cv2
import tqdm
import subprocess
import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt


class TestIOVideo(TestCase):
    def test_write_frames_preview(self):

        video_file = 'data/'
        filename = os.path.join(video_file, 'test_data.avi')
        frames = np.random.randint(0, 256, size=(300, 424, 512), dtype='int16')
        threads = 6
        fps = 30
        pixel_format = 'rgb24'
        codec = 'h264'
        slices=24
        slicecrc=1
        frame_size=None
        depth_min=0
        depth_max=80
        get_cmd=False
        cmap='jet'
        text=None
        text_scale=1
        text_thickness=2
        pipe=None
        close_pipe=True
        progress_bar=True

        if not np.mod(frames.shape[1], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

        if not np.mod(frames.shape[2], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)

        if not frame_size and type(frames) is np.ndarray:
            frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
        elif not frame_size and type(frames) is tuple:
            frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (255, 255, 255)
        txt_pos = (5, frames.shape[-1] - 40)

        command = ['ffmpeg',
                   '-y',
                   '-loglevel', 'fatal',
                   '-threads', str(threads),
                   '-framerate', str(fps),
                   '-f', 'rawvideo',
                   '-s', frame_size,
                   '-pix_fmt', pixel_format,
                   '-i', '-',
                   '-an',
                   '-vcodec', codec,
                   '-slices', str(slices),
                   '-slicecrc', str(slicecrc),
                   '-r', str(fps),
                   '-pix_fmt', 'yuv420p',
                   filename]

        if get_cmd:
            return command

        if not pipe:
            pipe = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # scale frames d00d

        use_cmap = plt.get_cmap(cmap)

        for i in tqdm.tqdm(range(frames.shape[0]), desc="Writing frames", disable=~progress_bar):
            disp_img = frames[i, ...].copy().astype('float32')
            disp_img = (disp_img-depth_min)/(depth_max-depth_min)
            disp_img[disp_img < 0] = 0
            disp_img[disp_img > 1] = 1
            disp_img = np.delete(use_cmap(disp_img), 3, 2)*255
            if text is not None:
                disp_img = cv2.putText(disp_img, text, txt_pos, font,
                                       text_scale, white, text_thickness, cv2.LINE_AA)
            pipe.stdin.write(disp_img.astype('uint8').tostring())

        if close_pipe:
            pipe.stdin.close()
            return None
        else:
            return pipe
