import os
import cv2
import unittest
import numpy as np
import networkx as nx
from unittest import TestCase
import matplotlib.pyplot as plt
from moseq2_viz.util import parse_index, read_yaml
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.viz import (clean_frames, make_crowd_matrix, position_plot, scalar_plot,
                            plot_syll_stats_with_sem, save_fig, plot_group_scalar_violin_plots)
from moseq2_viz.model.util import (parse_model_results, get_syllable_statistics,
                                   relabel_by_usage, get_syllable_slices, compute_behavioral_statistics)
from moseq2_viz.model.trans_graph import (convert_ebunch_to_graph, convert_transition_matrix_to_ebunch,
                                          get_transition_matrix, graph_transition_matrix)

def get_fake_movie():
    edge_size = 40
    points = np.arange(-edge_size, edge_size)
    sig1 = 10
    sig2 = 20

    kernel = np.exp(-(points ** 2.0) / (2.0 * sig1 ** 2.0))
    kernel2 = np.exp(-(points ** 2.0) / (2.0 * sig2 ** 2.0))

    kernel_full = np.outer(kernel, kernel2)
    kernel_full /= np.max(kernel_full)
    kernel_full *= 50

    fake_mouse = kernel_full
    fake_mouse[fake_mouse < 5] = 0

    tmp_image = np.ones((424, 512), dtype='int16') * 1000
    center = np.array(tmp_image.shape) // 2

    mouse_dims = np.array(fake_mouse.shape) // 2

    # put a mouse on top of a disk

    roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (300, 300)).astype('int16') * 300
    roi_dims = np.array(roi.shape) // 2

    tmp_image[center[0] - roi_dims[0]:center[0] + roi_dims[0],
    center[1] - roi_dims[1]:center[1] + roi_dims[1]] = \
        tmp_image[center[0] - roi_dims[0]:center[0] + roi_dims[0],
        center[1] - roi_dims[1]:center[1] + roi_dims[1]] - roi

    tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
    center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] = \
        tmp_image[center[0] - mouse_dims[0]:center[0] + mouse_dims[0],
        center[1] - mouse_dims[1]:center[1] + mouse_dims[1]] - fake_mouse

    fake_movie = np.tile(tmp_image, (20, 1, 1))
    return fake_movie

def get_ebunch(max_syllable=40, group=('Group1', 'default'), ret_trans=False):
    model_fit = 'data/test_model.p'
    index_file = 'data/test_index.yaml'
    config_file = 'data/config.yaml'

    index_data = read_yaml(index_file)
    index_data['pca_path'] = 'data/_pca/pca_scores.h5'

    config_data = read_yaml(config_file)

    anchor = 0

    model_data = parse_model_results(model_fit)
    index, sorted_index = parse_index(index_file)
    labels = model_data['labels']

    if 'train_list' in model_data.keys():
        label_uuids = model_data['train_list']
    else:
        label_uuids = model_data['keys']

    label_group = []

    print('Sorting labels...')

    if 'group' in index['files'][0].keys() and len(group) > 0:
        for uuid in label_uuids:
            label_group.append(sorted_index['files'][uuid]['group'])
    else:
        label_group = [''] * len(model_data['labels'])
        group = list(set(label_group))

    trans_mats = []
    usages = []
    for plt_group in group:
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        if use_labels:
            trans_mats.append(get_transition_matrix(use_labels, normalize=config_data['normalize'], combine=True,
                                                    max_syllable=max_syllable))
            usages.append(get_syllable_statistics(use_labels)[0])

    ngraphs = len(trans_mats)

    if anchor > ngraphs:
        print('Setting anchor to 0')
        anchor = 0

    ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(trans_mats[anchor], trans_mats[anchor],
                                                                 usages=usages[anchor])
    if not ret_trans:
        return ebunch_anchor, orphans
    else:
        return trans_mats, usages

class TestViz(TestCase):

    def test_save_fig(self):

        fig = plt.figure(1, figsize=(1,1))

        save_fig(fig, output_file='data/test_fig')
        assert os.path.exists('data/test_fig.png')
        assert os.path.exists('data/test_fig.pdf')
        os.remove('data/test_fig.png')
        os.remove('data/test_fig.pdf')

    def test_clean_frames(self):
        frames = get_fake_movie()
        medfilter_space = [0]
        gaussfilter_space = [0, 0]
        tail_filter = None

        out = clean_frames(frames, medfilter_space, gaussfilter_space, tail_filter)
        np.testing.assert_array_equal(frames, out)

        medfilter_space = [1]
        gaussfilter_space = [1.5, 1]

        out = clean_frames(frames, medfilter_space, gaussfilter_space, tail_filter)
        np.all(np.not_equal(frames, out))

        medfilter_space = [3]
        gaussfilter_space = [2.5, 2]

        out = clean_frames(frames, medfilter_space, gaussfilter_space, tail_filter)
        np.all(np.not_equal(frames, out))

        medfilter_space = [2]
        gaussfilter_space = [2.5, 2]
        tail_filter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        out = clean_frames(frames, medfilter_space, gaussfilter_space, tail_filter)
        np.all(np.not_equal(frames, out))

    def test_convert_transition_matrix_to_ebunch(self):
        max_syllable = 40
        ebunch_anchor, orphans = get_ebunch(max_syllable=max_syllable)

        assert all([isinstance(v, tuple) for v in ebunch_anchor]), "Ebunch return types != tuple"
        assert len(ebunch_anchor) == (max_syllable) * (max_syllable), \
            "Incorrect Number of transition matrix nodes"
        assert len(orphans) == 0, "Unwanted orphan nodes were generated"

    def test_convert_ebunch_to_graph(self):
        ebunch_anchor, _ = get_ebunch()
        g = convert_ebunch_to_graph(ebunch_anchor)
        assert isinstance(g, nx.DiGraph), "Return type is not a networkx Digraph"

    def test_graph_transition_matrix(self):
        group = ['Group1']
        trans_mats, _ = get_ebunch(group=group, ret_trans=True)
        plt, _, _ = graph_transition_matrix(trans_mats, groups=group)

        outfile = 'data/test_transition.png'
        plt.savefig(outfile)
        assert os.path.exists(outfile), "Transition graph was not saved."
        os.remove(outfile)

    def test_make_crowd_matrix(self):

        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i, _ in enumerate(index_data['files']):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
            index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        model_data = parse_model_results(model_fit)
        labels = model_data['labels']

        labels, _ = relabel_by_usage(labels)
        label_uuids = model_data['keys']

        syllable_slices = get_syllable_slices(2, labels, label_uuids, index_data)

        crowd_matrix = make_crowd_matrix(syllable_slices, rotate=True, center=True)
        print(crowd_matrix.shape)
        assert crowd_matrix.shape[0] == 120, "Crowd movie number of frames is incorrect"
        assert crowd_matrix.shape == (120, 424, 512), "Crowd movie resolution is incorrect"

        crowd_matrix = make_crowd_matrix(syllable_slices, max_dur=None, nexamples=1)
        assert crowd_matrix.shape[0] == 62, "Crowd movie number of frames is incorrect"
        assert crowd_matrix.shape == (62, 424, 512), "Crowd movie resolution is incorrect"

    def test_position_plot(self):
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
            index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        scalar_df = scalars_to_dataframe(index_data)
        plt, _, _ = position_plot(scalar_df)
        outfile = 'data/test_position.png'
        plt.savefig(outfile)

        assert os.path.exists(outfile), "Position graph was not saved."
        os.remove(outfile)

    def test_scalar_plot(self):
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i in range(len(index_data['files'])):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
            index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        scalar_df = scalars_to_dataframe(index_data)
        plt, _ = scalar_plot(scalar_df)
        outfile = 'data/test_scalars.png'
        plt.savefig(outfile)

        assert os.path.exists(outfile), "Scalars plot was not saved."
        os.remove(outfile)

    def test_plot_syll_stats_with_sem(self):
        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        _, sorted_index = parse_index(test_index)
        scalar_df = scalars_to_dataframe(sorted_index, model_path=test_model)

        complete_df = compute_behavioral_statistics(scalar_df)

        # mutation order plot with correct parameters
        fig, _ = plot_syll_stats_with_sem(complete_df, stat='usage', ordering='diff', max_sylls=None, groups=None,
                                       ctrl_group='default', exp_group='default', colors=['red', 'orange'])

        assert fig is not None

        # different stat selected, len(colors) < len(groups)
        fig, _ = plot_syll_stats_with_sem(complete_df, stat='duration', ordering='stat', max_sylls=40,
                                            groups=['Group1', 'Group2'], ctrl_group=None, exp_group=None, colors=['red'])

        assert fig is not None

        # incorrect groups, and empty colors, descending order sorting
        fig, _ = plot_syll_stats_with_sem(complete_df, stat='duration', ordering='stat', max_sylls=None,
                                       groups=['Group', 'Group2'], ctrl_group=None, exp_group=None, colors=[])

        assert fig is not None

        # currently raises error if user inputs incorrect ctrl_group/exp_group name
        fig, _ = plot_syll_stats_with_sem(complete_df, stat='usage', ordering='stat', max_sylls=None, groups=None,
                                       ctrl_group='Grou1', exp_group='Group2', colors=['red', 'orange'])

        assert fig is not None

    def test_plot_group_scalar_violin_plots(self):
        pass

if __name__ == '__main__':
    unittest.main()
