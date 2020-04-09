import os
import shutil
import joblib
import numpy as np
import ruamel.yaml as yaml
from unittest import TestCase
from moseq2_viz.util import check_video_parameters, parse_index
from moseq2_viz.model.util import parse_model_results, relabel_by_usage
from moseq2_viz.io.video import write_crowd_movies, write_frames_preview


class TestIOVideo(TestCase):

    def test_write_crowd_movies(self):

        index_file = 'data/test_index_crowd.yaml'
        model_path = 'data/mock_model.p'
        config_file = 'data/config.yaml'
        output_dir = 'data/crowd_movies/'
        max_syllable = 5
        max_examples = 40

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        f.close()

        if config_data['sort']:
            filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
        else:
            filename_format = 'syllable_{:d}.mp4'

        model_fit = parse_model_results(joblib.load(model_path))
        labels = model_fit['labels']

        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        index, sorted_index = parse_index(index_file)
        vid_parameters = check_video_parameters(sorted_index)
        clean_params = {
            'gaussfilter_space': config_data['gaussfilter_space'],
            'medfilter_space': config_data['medfilter_space']
        }

        if config_data['sort']:
            labels, ordering = relabel_by_usage(labels, count=config_data['count'])
        else:
            ordering = list(range(max_syllable))

        write_crowd_movies(sorted_index, config_data, filename_format, vid_parameters, clean_params, ordering,
                           labels, label_uuids, max_syllable, max_examples, output_dir)

        assert (os.path.exists(output_dir))
        assert (len(os.listdir(output_dir)) == max_syllable)
        shutil.rmtree(output_dir)

    def test_write_frames_preview(self):

        video_file = 'data/'
        filename = os.path.join(video_file, 'test_data2.avi')
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

        out = write_frames_preview(filename, frames, threads, fps, pixel_format, codec, slices, slicecrc,\
                             frame_size, depth_min, depth_max, get_cmd, cmap, text, text_scale, text_thickness,\
                             pipe, close_pipe, progress_bar)

        assert os.path.exists(filename)
        assert out == None
        os.remove(filename)