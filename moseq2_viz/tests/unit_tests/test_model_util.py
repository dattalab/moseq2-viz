import numpy as np
import pandas as pd
from operator import add
from functools import reduce
from unittest import TestCase
from moseq2_viz.model.util import (
    _get_transitions, calculate_syllable_usage, compress_label_sequence, find_label_transitions,
    get_syllable_statistics, parse_model_results
    )

def make_sequence(lbls, durs):
    arr = [[x] * y for x, y in zip(lbls, durs)]
    return np.array(reduce(add, arr))

class TestModelUtils(TestCase):

    def test_get_average_syllable_durations(self):
        print()

    def test_merge_models(self):
        print()

    def test_get_transitions(self):
        true_labels = [1, 2, 4, 1, 5]
        durs = [3, 4, 2, 6, 7]
        arr = make_sequence(true_labels, durs)

        trans, locs = _get_transitions(arr)

        assert true_labels[1:] == list(trans), 'syllable labels do not match with the transitions'
        assert list(np.diff(locs)) == durs[1:-1], 'syllable locations do not match their durations'

    def test_get_transition_martrix(self):
        print()

    def test_get_mouse_syllable_slices(self):
        print()

    def test_syllable_slices_from_dict(self):
        print()

    def test_get_syllable_slices(self):
        print()

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
        print()

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
        print()

    def test_labels_to_changepoints(self):
        print()

    def test_parse_batch_modeling(self):
        print()

    def test_parse_model_results(self):
        print()

    def test_relabel_by_usage(self):
        print()

    def test_results_to_dataframe(self):
        print()

    def test_model_datasets_to_df(self):
        print()

    def test_simulate_ar_trajectory(self):
        print()

    def test_sort_batch_results(self):
        print()

    def test_normalize_pcs(self):
        print()

    def test_gen_to_arr(self):
        print()

    def test_retrieve_pcs_from_slices(self):
        print()
