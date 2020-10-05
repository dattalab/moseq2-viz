import joblib
import numpy as np
from operator import add
from copy import deepcopy
import ruamel.yaml as yaml
from functools import reduce
from unittest import TestCase
import matplotlib.pyplot as plt
from collections import OrderedDict
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import parse_model_results
from moseq2_viz.model.trans_graph import get_pos, get_usage_dict, get_trans_graph_groups, \
    get_group_trans_mats, get_transition_matrix, graph_transition_matrix, get_stat_thresholded_ebunch, \
    _get_transitions, make_transition_graphs, make_difference_graphs, make_graph, draw_graph, normalize_matrix, \
    convert_ebunch_to_graph, convert_transition_matrix_to_ebunch, compute_and_graph_grouped_TMs, floatRgb, \
    threshold_edges, handle_graph_layout

def make_sequence(lbls, durs):
    arr = [[x] * y for x, y in zip(lbls, durs)]
    return np.array(reduce(add, arr))

class TestModelTransGraph(TestCase):

    def test_get_trans_graph_groups(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        assert group == ['Group1']
        assert label_group == ['Group1', 'Group1']
        assert label_uuids == ['5c72bf30-9596-4d4d-ae38-db9a7a28e912', 'abe92017-1d40-495e-95ef-e420b7f0f4b9']

    def test_get_group_trans_mats(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        assert len(trans_mats) == 1
        assert trans_mats[0].shape == (20, 20)
        assert len(usages) == 1
        assert len(usages[0]) == 20

    def test_compute_and_graph_TMs(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        config_file = 'data/config.yaml'
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')
        fig, _, _ = graph_transition_matrix(trans_mats,
                                      **config_data,
                                      usages=usages,
                                      groups=group,
                                      headless=True)

        test_fig = compute_and_graph_grouped_TMs(config_data, model['labels'], label_group, group)

        assert type(fig) == type(test_fig)

    def test__get_transitions(self):

        test_model = 'data/test_model.p'
        model = parse_model_results(joblib.load(test_model))

        transitions, locs = _get_transitions(model['labels'][0])

        assert len(transitions) == 859
        assert len(locs) == 859

        transitions, locs = _get_transitions(model['labels'][1])

        assert len(transitions) == 859
        assert len(locs) == 859

        true_labels = [1, 2, 4, 1, 5]
        durs = [3, 4, 2, 6, 7]
        arr = make_sequence(true_labels, durs)

        trans, locs = _get_transitions(arr)

        assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
        assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'

    def test_normalize_matrix(self):

        normalizations = ['bigram', 'rows', 'columns']
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20
        smoothing = 0.0

        init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing

        for v in model['labels']:
            # Get syllable transitions
            transitions = _get_transitions(v)[0]

            # Populate matrix array with transition data
            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

        for norm in normalizations:
            normed_mtx = normalize_matrix(deepcopy(init_matrix), norm)
            assert np.any(np.not_equal(init_matrix, normed_mtx))

    def test_get_transition_matrix(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20

        trans_mats = get_transition_matrix(model['labels'], max_syllable=max_syllable)

        assert len(trans_mats) == 2
        assert trans_mats[0].shape == (21, 21)

    def test_convert_ebunch_to_graph(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20
        edge_threshold = .0025

        tm = get_transition_matrix(model['labels'], max_syllable=max_syllable)[0]

        ebunch = convert_transition_matrix_to_ebunch(
            tm, tm, edge_threshold=edge_threshold, keep_orphans=False,
            max_syllable=tm.shape[0])[0]

        graph = convert_ebunch_to_graph(ebunch)

        assert graph.number_of_nodes() == 13

    def test_floatRgb(self):

        mag = 1
        cmin = 0
        cmax = 1

        r, g, b = floatRgb(mag, cmin, cmax)
        assert isinstance(r, float) and r <= 1
        assert isinstance(g, float) and g <= 1
        assert isinstance(b, float) and b <= 1

    def test_get_stat_thresholded_ebunch(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20

        tm = get_transition_matrix(model['labels'], max_syllable=max_syllable)[0]

        ebunch = convert_transition_matrix_to_ebunch(tm, tm,
                                                     edge_threshold=0,
                                                     keep_orphans=False,
                                                     max_syllable=tm.shape[0])[0]

        usages = get_group_trans_mats(model['labels'], ['Group 1', 'Group 1'], ['Group 1'], 21, normalize='bigram')[1][0]

        new_ebunch = get_stat_thresholded_ebunch(deepcopy(ebunch), usages, 0)
        assert new_ebunch == ebunch

    def test_threshold_edges(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20
        edge_threshold = .0025

        tm = get_transition_matrix(model['labels'], max_syllable=max_syllable)[0]

        ebunch, orphans = convert_transition_matrix_to_ebunch(
            tm, tm, keep_orphans=True, max_syllable=tm.shape[0])

        for i, v in np.ndenumerate(tm):
            new_ebunch, new_orphans = threshold_edges(deepcopy(ebunch), deepcopy(orphans), edge_threshold, tm, i, edge=v)

        assert ebunch == new_ebunch
        assert len(orphans) == len(new_orphans)-1

    def test_convert_transition_matrix_to_ebunch(self):

        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        max_syllable = 20
        edge_threshold = .0025

        tm = get_transition_matrix(model['labels'], max_syllable=max_syllable)[0]

        ebunch, orphans = convert_transition_matrix_to_ebunch(tm, tm,
                                                              edge_threshold=edge_threshold,
                                                              max_syllable=max_syllable)

        assert len(ebunch) == 14
        assert len(orphans) == 427

    def test_get_usage_dict(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')[1]

        usage_dict = get_usage_dict(usages)[0]

        assert isinstance(usage_dict, OrderedDict)
        assert len(usage_dict.keys()) == 20

    def test_handle_graph_layout(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        assert all([isinstance(usage_dict, OrderedDict) for usage_dict in usages]) == True
        assert anchor == 0
        assert usages_anchor == usages[0]
        assert ngraphs == 1

    def test_make_graph(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        edge_threshold = 0.0025

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages = get_usage_dict(usages)
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            trans_mats[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=0, max_syllable=20)

        graph = make_graph(trans_mats[anchor],
                           ebunch_anchor=ebunch_anchor,
                           edge_threshold=edge_threshold,
                           usages_anchor=usages_anchor)

        assert graph.number_of_nodes() == 12

    def test_make_difference_graph(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        edge_threshold = 0.0025

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)
        group += ['Group2']
        label_group[1] = 'Group2'
        group_names = group.copy()

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages = get_usage_dict(usages)
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            trans_mats[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=0, max_syllable=20)

        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
        nnodes = len(graph_anchor.nodes())
        pos = get_pos(graph_anchor, 'circular', nnodes)

        new_usages, new_group_names, new_difference_graphs, \
        new_widths, new_node_sizes, new_node_edge_colors, new_scalars = \
            make_difference_graphs(trans_mats, usages, group, group_names, usages_anchor,
                                   trans_mats, pos, ebunch_anchor, node_edge_colors=['r'])

        assert len(new_usages) == 3
        assert len(new_group_names) == 3
        assert len(new_difference_graphs) == 1 # no previously generated graphs were passed
        assert new_group_names[-1] == 'Group2 - Group1'

    def test_make_transition_graphs(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        edge_threshold = 0.0025

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)
        group += ['Group2']
        label_group[1] = 'Group2'
        group_names = group.copy()

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages = get_usage_dict(usages)
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            trans_mats[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=0, max_syllable=20)

        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
        nnodes = len(graph_anchor.nodes())
        pos = get_pos(graph_anchor, 'circular', nnodes)

        usages, group_names, widths, node_sizes, node_edge_colors, graphs, scalars = \
            make_transition_graphs(trans_mats, usages, group, group_names, usages_anchor,
                                   pos, ebunch_anchor, orphans, edge_threshold=.0025,
                                   difference_threshold=.0005, orphan_weight=0)

        assert len(usages) == 3
        assert len(group_names) == 3
        assert len(widths) == 3
        assert len(node_sizes) == 3
        assert len(node_edge_colors) == 3
        assert len(graphs) == 3
        assert scalars == None

    def test_get_pos(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        edge_threshold = 0.0025

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages = get_usage_dict(usages)
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            trans_mats[anchor], trans_mats[anchor],
            edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=0, max_syllable=20)

        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
        nnodes = graph_anchor.number_of_nodes()

        layouts = ['spring', 'circular', 'spectral', usages_anchor]
        for layout in layouts:
            pos = get_pos(graph_anchor, layout, nnodes)
            assert isinstance(pos, dict)
            if isinstance(layout, str):
                assert len(pos.keys()) == nnodes
            else:
                assert len(pos.keys()) == len(usages_anchor.keys())

    def test_draw_graphs(self):
        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        width_per_group = 8
        edge_threshold = 0.0025

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)
        group += ['Group2']
        label_group[1] = 'Group2'
        group_names = group.copy()

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        usages = get_usage_dict(usages)
        usages, anchor, usages_anchor, ngraphs = handle_graph_layout(trans_mats, usages, anchor=0)

        ebunch_anchor, orphans = convert_transition_matrix_to_ebunch(
            trans_mats[anchor], trans_mats[anchor], edge_threshold=edge_threshold,
            keep_orphans=True, usages=usages_anchor,
            usage_threshold=0, max_syllable=20)

        graph_anchor = convert_ebunch_to_graph(ebunch_anchor)
        nnodes = len(graph_anchor.nodes())
        pos = get_pos(graph_anchor, 'circular', nnodes)

        usages, group_names, widths, node_sizes, node_edge_colors, graphs, scalars = \
            make_transition_graphs(trans_mats, usages, group, group_names, usages_anchor,
                                   pos, ebunch_anchor, orphans, edge_threshold=.0025,
                                   difference_threshold=.0005, orphan_weight=0)

        fig, ax = plt.subplots(ngraphs, ngraphs,
                               figsize=(ngraphs * width_per_group,
                                        ngraphs * width_per_group))

        for i, graph in enumerate(graphs):
            if i == ngraphs:
                i = 0
                j = 1
            draw_graph(graph, group_names[i], widths[i], pos, node_color='w',
                       node_size=node_sizes[i], node_edge_colors=node_edge_colors[i],
                       ax=ax, i=i, j=i)

        assert fig != None
        assert ax.shape == (2, 2)

    def test_graph_transition_matrix(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        model = parse_model_results(joblib.load(test_model))
        index, sorted_index = parse_index(test_index)

        group, label_group, label_uuids = get_trans_graph_groups(model, index, sorted_index)
        group += ['Group2']
        label_group[1] = 'Group2'

        trans_mats, usages = get_group_trans_mats(model['labels'], label_group, group, 20, normalize='bigram')

        fig, ax, pos = graph_transition_matrix(trans_mats, usages=usages, groups=group)

        assert len(pos.keys()) == 12
        assert ax.shape == (2, 2)
        assert fig != None
