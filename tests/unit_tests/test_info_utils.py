import joblib
import numpy as np
from unittest import TestCase
from moseq2_viz.model.util import parse_model_results
from moseq2_viz.info.util import entropy, entropy_rate
from moseq2_viz.model.util import get_syllable_statistics, get_transition_matrix


class TestInfoUtils(TestCase):
    def test_entropy(self):
        model_fit = 'data/test_model.p'

        model_data = parse_model_results(joblib.load(model_fit))
        labels = model_data['labels']
        truncate_syllable = 40
        smoothing = 1.0

        ent = []
        for v in labels:
            usages = get_syllable_statistics([v])[0]
            assert usages != None

            syllables = np.array(list(usages.keys()))
            truncate_point = np.where(syllables == truncate_syllable)[0]

            if truncate_point is None or len(truncate_point) != 1:
                truncate_point = len(syllables)
            else:
                truncate_point = truncate_point[0]

            assert truncate_point > 0

            usages = np.array(list(usages.values()), dtype='float')
            usages = usages[:truncate_point] + smoothing
            usages /= usages.sum()

            ent.append(-np.sum(usages * np.log2(usages)))

        test_ent = entropy(labels)

        assert len(test_ent) == 2 # for 2 sessions in modeling
        np.testing.assert_almost_equal(ent, test_ent, 1)

    def test_entropy_rate(self):

        model_fit = 'data/test_model.p'

        model_data = parse_model_results(joblib.load(model_fit))
        labels = model_data['labels']
        truncate_syllable = 100
        smoothing = 1.0
        tm_smoothing = 1.0

        for v in labels:

            usages = get_syllable_statistics([v])[0]
            syllables = np.array(list(usages.keys()))
            truncate_point = np.where(syllables == truncate_syllable)[0]

            if truncate_point is None or len(truncate_point) != 1:
                truncate_point = len(syllables)
            else:
                truncate_point = truncate_point[0]

            usages = np.array(list(usages.values()), dtype='float')
            usages = usages[:truncate_point] + smoothing
            usages /= usages.sum()

            for norm in ['none', 'bigram', 'rows', 'columns']:

                tm = get_transition_matrix([v],
                                           max_syllable=100,
                                           normalize='none',
                                           smoothing=0.0,
                                           disable_output=True)[0] + tm_smoothing
                tm = tm[:truncate_point, :truncate_point]

                assert tm.shape == (truncate_point, truncate_point)

                if norm == 'bigram':
                    tm /= tm.sum()
                elif norm == 'rows':
                    tm /= tm.sum(axis=1, keepdims=True)
                elif norm == 'columns':
                    tm /= tm.sum(axis=0, keepdims=True)

                real_er = -np.sum(usages[:, None] * tm * np.log2(tm))

                test_er = entropy_rate(labels, normalize=norm, truncate_syllable=truncate_syllable,
                                       smoothing=smoothing, tm_smoothing=tm_smoothing)

                assert len(test_er) == 2  # for 2 sessions in modeling

                np.testing.assert_allclose(real_er, test_er, rtol=1e-2)