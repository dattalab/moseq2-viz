'''

Helper functions for handling crowd movie file writing and video metadata maintenance.

'''

import os
import cv2
import warnings
import subprocess
import numpy as np
import ruamel.yaml as yaml
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from moseq2_viz.viz import make_crowd_matrix
from moseq2_viz.util import check_video_parameters
from moseq2_viz.model.util import get_syllable_slices

def get_video_info(filename):
    '''

    Get dimensions of data compressed using ffv1, along with duration via ffmpeg.

    Parameters
    ----------
    filename (string): name of file

    Returns
    -------
    (dict): dictionary containing video file metadata
    '''

    command = ['ffprobe',
               '-v', 'fatal',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames',
               '-of',
               'default=noprint_wrappers=1:nokey=1',
               filename,
               '-sexagesimal']

    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()

    if(err):
        print(err)
    out = out.decode().split('\n')
    try:
        return {'file': filename,
            'dims': (int(float(out[0])), int(float(out[1]))),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'nframes': int(out[3])}
    except:
        print('Could not process this video extension:', filename)
        return {}

def write_crowd_movie_info_file(model_path, model_fit, index_file, output_dir):
    '''
    Creates an info.yaml file in the crowd movie directory that holds model training parameters.
    This file helps identify the conditions from which the crowd movies were generated.
    Parameters
    ----------
    model_path (str): path to model used to generate movies
    model_fit (dict): loaded ARHMM dict
    index_file (str): path to index file used with model
    output_dir (str): path to crowd movies directory to store file in.
    Returns
    -------
    None
    '''

    # Crowd movie info file contents; used to indicate the modeling state the crowd_movies were generated from
    info_parameters = ['model_class', 'kappa', 'gamma', 'alpha']

    # Loading parameters to dict to save to file in output directory
    info_file = os.path.join(output_dir, 'info.yaml')
    info_dict = {k: model_fit['model_parameters'][k] for k in info_parameters}

    # Adding model and index file paths
    info_dict['model_path'] = model_path
    info_dict['index_path'] = index_file

    # Convert numpy dtypes to their corresponding primitives
    for k, v in info_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            info_dict[k] = info_dict[k].item()

    # Write metadata info file
    with open(info_file, 'w+') as f:
        yaml.safe_dump(info_dict, f)

def write_crowd_movies(sorted_index, config_data, ordering, labels, label_uuids, output_dir):
    '''
    Creates syllable slices for crowd movies and writes them to files.
    Parameters
    ----------
    sorted_index (dict): dictionary of sorted index data.
    config_data (dict): dictionary of visualization parameters.
    filename_format (str): string format that denotes the saved crowd movie file names.
    ordering (list): ordering for the new mapping of the relabeled syllable usages.
    labels (numpy ndarray): list of syllable usages
    label_uuids (list): list of session uuids each series of labels belongs to.
    output_dir (str): path directory where all the movies are written.
    Returns
    -------
    None
    '''

    # Filtering parameters
    clean_params = {
        'gaussfilter_space': config_data['gaussfilter_space'],
        'medfilter_space': config_data['medfilter_space']
    }

    # Set crowd movie filename format based on whether syllables were relabeled
    if config_data['sort']:
        filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
    else:
        filename_format = 'syllable_{:d}.mp4'

    # Ensure all video metadata parameters are consistent
    vid_parameters = check_video_parameters(sorted_index)
    if vid_parameters['resolution'] is not None:
        config_data['raw_size'] = vid_parameters['resolution']

    with mp.Pool() as pool:
        # Get frame slices from all included sessions for each syllable label
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index=sorted_index)
        
        with warnings.catch_warnings():
            slices = list(tqdm(pool.imap(slice_fun, config_data['crowd_syllables']),
                               total=config_data['max_syllable'], desc='Getting Syllable Slices', 
                               disable=not config_data['progress_bar']))

        matrix_fun = partial(make_crowd_matrix,
                             nexamples=config_data['max_examples'],
                             dur_clip=config_data['dur_clip'],
                             min_height=config_data['min_height'],
                             crop_size=vid_parameters['crop_size'],
                             raw_size=config_data['raw_size'],
                             scale=config_data['scale'],
                             legacy_jitter_fix=config_data['legacy_jitter_fix'],
                             **clean_params)

        # Compute crowd matrices
        with warnings.catch_warnings():
            # creating crowd matrices
            crowd_matrices = list(tqdm(pool.imap(matrix_fun, slices), total=len(config_data['crowd_syllables']),
                                       desc='Getting Crowd Matrices', disable=not config_data['progress_bar']))
        
        # writing function
        config_data['fps'] = vid_parameters['fps']
        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=config_data['min_height'],
                            depth_max=config_data['max_height'], cmap=config_data['cmap'], progress_bar = config_data['progress_bar'])

        # get list of tuples (path_to_write, crowd_movie)
        crowd_movies = [[os.path.join(output_dir, filename_format.format(i, config_data['count'], ordering[i])), crowd_matrix]
                            for i, crowd_matrix in tqdm(enumerate(crowd_matrices), total=len(config_data['crowd_syllables']), 
                                                        desc='Writing Movies', disable=not config_data['progress_bar']) 
                            if crowd_matrix is not None]

        if len(config_data['crowd_syllables']) > 1:
            crowd_movie_paths = [cm[0] for cm in crowd_movies]
        else:
            crowd_movies[0][0] = crowd_movies[0][0].replace('-0 ', f'-{config_data["crowd_syllables"][0]}')
            crowd_movies[0][0] = crowd_movies[0][0].replace(f'-{ordering[0]}.', f'-{ordering[config_data["crowd_syllables"][0]]}.')
            crowd_movie_paths = [cm[0] for cm in crowd_movies]

        # write movie
        pool.starmap(write_fun, crowd_movies)
        
    return crowd_movie_paths

def write_frames_preview(filename, frames=np.empty((0,)), threads=6,
                         fps=30, pixel_format='rgb24',
                         codec='h264', slices=24, slicecrc=1,
                         frame_size=None, depth_min=0, depth_max=80,
                         get_cmd=False, cmap='jet', text=None, text_scale=1,
                         text_thickness=2, pipe=None, close_pipe=True, progress_bar=True, **kwargs):
    '''
    Writes out a false-colored mp4 video.
    [Duplicate from moseq2-extract]

    Parameters
    ----------
    filename (str):
    frames (3D numpy array): num_frames * r * c
    threads (int): number of threads to write file
    fps (int): frames per second
    pixel_format (str): ffmpeg image formatting flag.
    codec (str): ffmpeg image encoding flag.
    slices (int): number of slices per thread.
    slicecrc (int): check integrity of slices.
    frame_size (tuple): image dimensions
    depth_min (int): minimum mouse distance from bucket floor
    depth_max (int): maximum mouse distance from bucket floor
    get_cmd (bool): return ffmpeg command instead of executing the command in python.
    cmap (str): color map selection.
    text (range(num_frames): display frame number in output video.
    text_scale (int): text size.
    text_thickness (int): text thickness.
    pipe (subProcess.Pipe object): if not None, indicates that there are more frames to be written.
    close_pipe (bool): indicates whether video is done writing, and to close pipe to file-stream.
    progress_bar (bool): display progress bar.
    kwargs (dict): extra keyword arguments

    Returns
    -------
    (subProcess.Pipe object): if there are more slices/chunks to write to, otherwise None.
    '''

    # Set frame padding
    if not np.mod(frames.shape[1], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

    if not np.mod(frames.shape[2], 2) == 0:
        frames = np.pad(frames, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)

    # Get string frame dimensions
    if not frame_size and type(frames) is np.ndarray:
        frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    elif not frame_size and type(frames) is tuple:
        frame_size = '{0:d}x{1:d}'.format(frames[0], frames[1])

    # Set text metadata to write frame numbers
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

    # Run ffmpeg command
    if not pipe:
        pipe = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Get color map
    use_cmap = plt.get_cmap(cmap)

    # Write movie
    for i in tqdm(range(frames.shape[0]), desc="Writing frames", disable=~progress_bar):
        disp_img = frames[i, :].copy().astype('float32')
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
