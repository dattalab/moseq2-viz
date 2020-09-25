import joblib
import unittest
import numpy as np
import pandas as pd
from operator import add
from copy import deepcopy
import ruamel.yaml as yaml
from functools import reduce
from unittest import TestCase
from moseq2_viz.model.trans_graph import _get_transitions, get_transition_matrix
from moseq2_viz.model.util import (relabel_by_usage, h5_to_dict,
    calculate_syllable_usage, compress_label_sequence, find_label_transitions,
    get_syllable_statistics, parse_model_results, merge_models, get_mouse_syllable_slices,
    syllable_slices_from_dict, get_syllable_slices, calculate_label_durations, labels_to_changepoints,
    results_to_dataframe, _gen_to_arr, normalize_pcs, _whiten_all, simulate_ar_trajectory)

def make_sequence(lbls, durs):
    arr = [[x] * y for x, y in zip(lbls, durs)]
    return np.array(reduce(add, arr))

class TestModelUtils(TestCase):

    def test_merge_models(self):
        model_paths = 'data/'
        ext = 'p'
        model_data = merge_models(model_paths, ext)

        assert len(model_data.keys()) > 0
        assert len(model_data['keys']) == 2

    def test_get_transitions(self):
        true_labels = [1, 2, 4, 1, 5]
        durs = [3, 4, 2, 6, 7]
        arr = make_sequence(true_labels, durs)

        trans, locs = _get_transitions(arr)

        assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
        assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'

    def test_get_transition_martrix(self):
        model_fit = 'data/test_model.p'

        model_data = parse_model_results(joblib.load(model_fit))
        labels = model_data['labels']
        max_syllable = 40
        normalize = 'bigram'
        smoothing = 0.0
        combine = False

        trans_mats = get_transition_matrix(labels, max_syllable, normalize, smoothing, combine)

        assert len(trans_mats) == 2 # number of sessions
        assert np.asarray(trans_mats).shape == (2, max_syllable+1, max_syllable+1) # ensuring square matrices

        normalize = 'rows'
        trans_mats = get_transition_matrix(labels, max_syllable, normalize, smoothing, combine)

        assert len(trans_mats) == 2  # number of sessions
        assert np.asarray(trans_mats).shape == (2, max_syllable + 1, max_syllable + 1)  # ensuring square matrices

        normalize = 'columns'
        trans_mats = get_transition_matrix(labels, max_syllable, normalize, smoothing, combine)

        assert len(trans_mats) == 2  # number of sessions
        assert np.asarray(trans_mats).shape == (2, max_syllable + 1, max_syllable + 1)  # ensuring square matrices

    def test_get_mouse_syllable_slices(self):
        syllable = 2
        labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2]
        slices = get_mouse_syllable_slices(syllable, labels)

        assert not isinstance(slices, list)
        assert slices != []
        assert list(slices) == [slice(3, 6, None), slice(9, 12, None)]

    def test_syllable_slices_from_dict(self):
        model_fit = 'data/test_model.p'
        index_file = 'data/test_index_crowd.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        lbl_dict = {}
        labels, _ = relabel_by_usage(model_data['labels'])
        for k,v in zip(model_data['keys'], labels):
            lbl_dict[k] = v

        ret = syllable_slices_from_dict(2, lbl_dict, index_data)

        assert isinstance(ret, dict)
        assert len(ret.keys()) == 2
        for i in range(2):
            assert len(list(ret.values())[i]) > 0
            assert np.nan not in list(ret.values())[i]


    def test_get_syllable_slices(self):
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        labels = model_data['labels']

        labels, _ = relabel_by_usage(labels)
        label_uuids = model_data['keys']

        lbl_dict = {}
        for k, v in zip(model_data['keys'], model_data['labels']):
            lbl_dict[k] = v

        ret = syllable_slices_from_dict(2, lbl_dict, index_data)
        syllable_slices = get_syllable_slices(2, labels, label_uuids, index_data)

        assert np.asarray(syllable_slices).shape == (2*len(list(ret.values())[0]), 3)
        assert len(syllable_slices) == 2*len(list(ret.values())[0])
        assert len(list(ret.values())[0]) == len(list(ret.values())[1])


    def test_find_label_transitions(self):
        lbls = [-5, 1, 3, 1, 4]
        durs = [3, 4, 10, 4, 12]
        arr = make_sequence(lbls, durs)

        inds = find_label_transitions(arr)

        assert list(inds) == list(np.cumsum(durs[:-1])), 'label indices do not align'


    def test_compress_label_sequence(self):
        lbls = [-5, 1, 3, 1, 4]
        durs = [3, 4, 10, 4, 12]
        arr = make_sequence(lbls, durs)

        compressed = compress_label_sequence(arr)

        assert lbls[1:] == list(compressed), 'compressed sequence does not match original'

    def test_calculate_label_durations(self):
        labels = [-5, -5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 5]
        change = np.diff(labels) != 0
        indices = np.where(change)[0]
        indices += 1
        indices = np.concatenate(([0], indices))

        onsets = np.concatenate((indices, [np.asarray(labels).size]))
        durations = np.diff(onsets)

        durs = calculate_label_durations(np.asarray(labels))

        assert all(durations[1:] == durs)

    def test_calculate_syllable_usage(self):
        true_labels = [-5, 1, 3, 1, 2, 4, 1, 5]
        durs = [3, 3, 4, 2, 6, 7, 2, 4]
        arr = make_sequence(true_labels, durs)

        labels_dict = {
            'animal1': arr,
            'animal2': arr
        }

        test_result = {
            1: 6,
            2: 2,
            3: 2,
            4: 2,
            5: 2
        }

        result = calculate_syllable_usage(labels_dict)

        assert test_result == result, 'syll usage calculation incorrect for dict'

        df = pd.DataFrame({'syllable': true_labels[1:] + true_labels[1:], 'dur': durs[1:] + durs[1:]})
        result = calculate_syllable_usage(df)

        assert test_result == result, 'syll usage calculation incorrect for dataframe'

    def test_get_syllable_statistics(self):
        # For now this just tests if there are any function-related errors
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        labels = model_data['labels']

        labels, _ = relabel_by_usage(labels)
        max_syllable = 40
        usages, durations = get_syllable_statistics(labels, max_syllable=max_syllable)
        assert len(usages) == len(durations) == max_syllable

        Fusages, Fdurations = get_syllable_statistics(labels, max_syllable=max_syllable, count='frames')
        assert len(Fusages) == len(Fdurations) == max_syllable

        assert usages.values() != Fusages.values()

        Uusages, Udurations = get_syllable_statistics(labels, max_syllable=max_syllable, count='fraasdfsdgmes')
        assert len(Uusages) == len(Udurations) == max_syllable

        assert list(usages.values()) == list(Uusages.values())

    def test_labels_to_changepoints(self):

        labels = np.asarray([[-5, -5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 5]])
        cps = labels_to_changepoints(labels, fs=30.)

        # number of changepoints == number of unique labels in list -1
        # number of unique labels is decremented by 1 to account for nlag prefixes
        assert len(cps) == len(list(set(labels[0])))-2

    def test_parse_model_results(self):
        model_fit = 'data/mock_model.p'

        model_dict = parse_model_results(model_fit)
        assert 'keys' in model_dict.keys()
        assert 'labels' in model_dict.keys()
        assert 'train_list' in model_dict.keys()

        model_dict2 = parse_model_results(model_fit, sort_labels_by_usage=True)
        assert 'keys' in model_dict2.keys()
        assert 'labels' in model_dict2.keys()
        assert 'train_list' in model_dict2.keys()

        model_dict3 = parse_model_results(model_fit, sort_labels_by_usage=True, map_uuid_to_keys=True)
        assert 'keys' in model_dict3.keys()
        assert 'labels' in model_dict3.keys()
        assert 'train_list' in model_dict3.keys()

        labels, _ = relabel_by_usage(model_dict['labels'])
        np.testing.assert_array_equal(model_dict2['labels'], labels)
        np.testing.assert_array_equal(model_dict2['labels'], list(model_dict3['labels'].values()))

    def test_relabel_by_usage(self):
        labels = np.asarray([[-5, -5, -5, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 5],
                             [-5, -5, -5, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 5, 5, 5, 5]])
        rel, ordering = relabel_by_usage(labels)

        actual_labels = []
        for arr in rel:
            subarr = []
            for i in arr:
                actual = ordering[i]
                subarr.append(actual)
            actual_labels.append(subarr)

        np.testing.assert_array_equal(actual_labels, labels)

    def test_results_to_dataframe(self):
        model_fit = 'data/test_model.p'
        index_file = 'data/test_index.yaml'
        max_syllable = 40

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        model_dict = parse_model_results(model_fit)
        df, label_df = results_to_dataframe(model_dict, index_data, sort=True, max_syllable=max_syllable, compute_labels=True)

        assert isinstance(label_df, pd.DataFrame)
        assert isinstance(df, pd.DataFrame)

        columns = ['duration', 'usage', 'uuid', 'group', 'syllable', 'SessionName', 'SubjectName', 'StartTime']
        assert set(columns) == set(df.columns)
        assert df.shape == ((max_syllable)*2, len(columns))

    def test_normalize_pcs(self):
        index_file = 'data/test_index.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')
        norm = normalize_pcs(pca_scores)

        print(pca_scores, norm)
        assert pca_scores.keys() == norm.keys()
        assert pca_scores.values() != norm.values()

        norm_scores = deepcopy(pca_scores)
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig

        assert norm_scores.keys() == norm.keys()
        np.testing.assert_array_almost_equal(np.asarray(list(norm_scores.values())), np.asarray(list(norm.values())))

        norm2 = normalize_pcs(pca_scores, 'm')
        norm_scores = deepcopy(pca_scores)
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = v - mu

        assert norm_scores.keys() == norm2.keys()
        np.testing.assert_array_almost_equal(np.asarray(list(norm_scores.values())), np.asarray(list(norm2.values())))

        norm3 = normalize_pcs(pca_scores, 'ind-zscore')
        norm_scores = deepcopy(pca_scores)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - np.nanmean(v)) / np.nanstd(v)

        assert norm_scores.keys() == norm3.keys()
        np.testing.assert_array_almost_equal(np.asarray(list(norm_scores.values())), np.asarray(list(norm3.values())))

        norm4 = normalize_pcs(pca_scores, 'ind-zscor12344e')
        norm_scores = deepcopy(pca_scores)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - np.nanmean(v)) / np.nanstd(v)

        assert norm_scores.keys() == norm4.keys()
        np.testing.assert_array_almost_equal(np.asarray(list(norm.values())), np.asarray(list(norm4.values())))

    def test_gen_to_arr(self):
        syllable = 2
        labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2]
        slices = get_mouse_syllable_slices(syllable, labels)
        out = _gen_to_arr(slices)
        assert type(out) == type(np.array([]))

    def test_whiten_all(self):
        index_file = 'data/test_index.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        pca_scores = h5_to_dict(index_data['pca_path'], 'scores')
        whitened_test = _whiten_all(pca_scores)
        assert pca_scores.values() != whitened_test.values()

        whitened_test2 = _whiten_all(pca_scores, center=False)

        assert pca_scores.values() != whitened_test.values() != whitened_test2.values()


    def test_simulate_ar_trajectory(self):
        model_path = 'data/mock_model.p'

        model_data = parse_model_results(joblib.load(model_path))
        ar_mats = np.array(model_data['model_parameters']['ar_mat'])

        sim_mats = simulate_ar_trajectory(ar_mats)
        assert sim_mats.shape == (100, 100)
        assert ar_mats.all() != sim_mats.all()


if __name__ == '__main__':
    unittest.main()