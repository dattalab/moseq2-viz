import numpy as np
import pandas as pd
from unittest import TestCase
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import results_to_dataframe
from moseq2_viz.model.label_util import syll_onset, syll_duration, syll_id, to_df, \
    get_syllable_mutation_ordering, get_sorted_syllable_stat_ordering

class TestTrainLabelUtils(TestCase):

    def test_syll_onset(self):

        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        change = np.diff(labels) != 0
        indices = np.where(change)[0]
        indices += 1
        indices = np.concatenate(([0], indices))
        assert all(indices == [0, 3, 8, 11, 14])
        assert all(indices == syll_onset(labels))


    def test_syll_duration(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        change = np.diff(labels) != 0
        indices = np.where(change)[0]
        indices += 1
        indices = np.concatenate(([0], indices))

        onsets = np.concatenate((indices, [np.asarray(labels).size]))
        durations = np.diff(onsets)
        assert all(durations == [3, 5, 3, 3, 4])
        assert all(durations == syll_duration(np.asarray(labels)))

    def test_syll_id(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        output = [0, 1, 2, 3, 0]

        assert all(syll_id(np.asarray(labels)) == output)

    def test_to_df(self):
        labels = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]
        pd_labels = pd.Series(labels)

        out1 = to_df(labels, ('id1'))
        out2 = to_df(pd_labels, ('id1'))

        assert (isinstance(out1, pd.DataFrame))
        assert (isinstance(out2, pd.DataFrame))

    def test_get_syllable_muteness_ordering(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        _, sorted_index = parse_index(test_index)

        ctrl_group = 'Group1'
        exp_group = 'Group2'

        for i, (k,v) in enumerate(sorted_index['files'].items()):
            if i == 1:
                sorted_index['files'][k]['group'] = 'Group2'

        complete_df, _ = results_to_dataframe(test_model, sorted_index)

        new_ordering = get_syllable_mutation_ordering(complete_df, ctrl_group, exp_group, max_sylls=None, stat='usage')
        ordering = get_sorted_syllable_stat_ordering(complete_df, stat='usage')

        assert list(new_ordering) != list(range(40))
        assert list(new_ordering) != list(ordering)


    def test_get_sorted_syllable_stat_ordering(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        _, sorted_index = parse_index(test_index)

        for i, (k, v) in enumerate(sorted_index['files'].items()):
            if i == 1:
                sorted_index['files'][k]['group'] = 'Group2'

        complete_df, _ = results_to_dataframe(test_model, sorted_index)

        ordering, relabel_mapping = get_sorted_syllable_stat_ordering(complete_df, stat='usage')

        assert list(relabel_mapping.keys()) == list(ordering)
        assert isinstance(relabel_mapping, dict)
        assert len(list(relabel_mapping.keys())) == 40
        assert list(ordering) != list(range(40))
