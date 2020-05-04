from functools import partial
from sys import platform
import click
import os
import ruamel.yaml as yaml
import h5py
import multiprocessing as mp
import numpy as np
import joblib
import tqdm
import warnings
import re
import shutil
import time
import psutil
import pytest
from click.testing import CliRunner
from moseq2_viz.cli import *

def test_add_group():
    input_dir = 'tests/test_files/'
    input_path = os.path.join(input_dir,'test_index.yaml')

    runner = CliRunner()

    group_params = ['-k', 'SubjectName',
                    '-v', 'Mouse',
                    '-g', 'Group1',
                    '-e', # FLAG
                    '--lowercase', # FLAG
                    '-n', # FLAG
                    input_path]

    results = runner.invoke(add_group, group_params)
    assert(not os.path.samefile(os.path.join(input_dir, 'orig.txt'), input_path))
    assert(results.exit_code == 0)

def copy_h5_metadata_to_yaml():
    input_dir = 'tests/test_files/'

    runner = CliRunner()
    results = runner.invoke(copy_h5_metadata_to_yaml, ['--h5-metadata-path', input_dir+'pca_scores.h5',
                                                       '-i', input_dir])

    assert (results.exit_code == 0)

def test_plot_scalar_summary():

    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'
    runner = CliRunner()

    results = runner.invoke(plot_scalar_summary, ['--output-file', gen_dir+'scalar',
                                                  input_dir+'test_index.yaml'])

    assert (os.path.exists(gen_dir + 'scalar_position.png'))
    assert (os.path.exists(gen_dir + 'scalar_position.pdf'))
    assert (os.path.exists(gen_dir + 'scalar_summary.png'))
    assert (os.path.exists(gen_dir + 'scalar_summary.pdf'))
    os.remove(gen_dir+'scalar_position.png')
    os.remove(gen_dir + 'scalar_position.pdf')
    os.remove(gen_dir + 'scalar_summary.png')
    os.remove(gen_dir + 'scalar_summary.pdf')
    assert (results.exit_code == 0)

def test_plot_transition_graph():

    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'
    runner = CliRunner()

    trans_params = ['--output-file', gen_dir+'transitions',
                    '--max-syllable', 40,
                    '-g', 'Group1',
                    #'--keep-orphans', True,
                    #'--orphan-weight', 0,
                    '--arrows', # FLAG
                    '--normalize', 'bigram',
                    '--layout', 'spring',
                    '--sort', True,
                    '--count', 'usage',
                    '--edge-scaling', 250,
                    '--node-scaling', 1e4,
                    '--scale-node-by-usage', True,
                    '--width-per-group', 8,
                    input_dir+'test_index.yaml',
                    input_dir+'mock_model.p']

    results = runner.invoke(plot_transition_graph, trans_params)

    assert(os.path.exists(gen_dir+'transitions.png'))
    assert (os.path.exists(gen_dir + 'transitions.pdf'))
    os.remove(gen_dir + 'transitions.png')
    os.remove(gen_dir + 'transitions.pdf')
    assert (results.exit_code == 0)

def test_plot_usages():
    input_dir = 'tests/test_files/'
    gen_dir = 'tests/test_files/gen_plots/'

    runner = CliRunner()

    use_params = ['--output-file', gen_dir+'test_usages',
                  '--sort', True,
                  '--count', 'usage',
                  '--max-syllable', 40,
                  '-g', 'Group1',
                  input_dir+'test_index.yaml',
                  input_dir+'mock_model.p']

    results = runner.invoke(plot_usages, use_params)

    assert (os.path.exists(gen_dir + 'test_usages.png'))
    assert (os.path.exists(gen_dir + 'test_usages.pdf'))
    os.remove(gen_dir + 'test_usages.png')
    os.remove(gen_dir + 'test_usages.pdf')
    assert (results.exit_code == 0)

def test_make_crowd_movies():
    input_dir = 'tests/test_files/'
    crowd_dir = input_dir+'crowd_movies/'
    max_examples = 40
    runner = CliRunner()

    crowd_params = ['-o', crowd_dir,
                    '--max-syllable', 40,
                    #'--t', 2,
                    #'--sort', True,
                    '--count', 'frames',
                    #'--gaussfilter-space', 0, 0,
                    #'--medfilter-space', 0,
                    '--min-height', 5,
                    '--max-height', 80,
                    '--raw-size', 512, 424,
                    '--scale', 1,
                    '--cmap', 'jet',
                    '--dur-clip', 300,
                    '--legacy-jitter-fix', False,
                    '--max-examples', max_examples,
                    input_dir+'test_index.yaml',
                    input_dir+'mock_model.p']

    results = runner.invoke(make_crowd_movies, crowd_params)

    assert(os.path.exists(crowd_dir))
    assert(len([os.listdir(crowd_dir)][0]) == max_examples+1)
    shutil.rmtree(crowd_dir)
    assert (results.exit_code == 0)

