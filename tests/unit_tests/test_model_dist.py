import numpy as np
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import parse_model_results, h5_to_dict
from moseq2_viz.model.dist import get_behavioral_distance, get_behavioral_distance_ar, \
    get_init_points, reformat_dtw_distances
from unittest import TestCase

class TestModeLDists(TestCase):

    def test_get_behavioral_distance(self):
        test_index = 'data/test_index_crowd.yaml'
        test_model = 'data/mock_model.p'

        index, sorted_index = parse_index(test_index)

        sorted_index['pca_path'] = 'data/test_scores.h5'

        dist_dict = get_behavioral_distance(sorted_index, test_model,
                                            max_syllable=10,
                                            resample_idx=-1,
                                            distances=['ar[init]', 'ar[dtw]'])

        assert len(dist_dict.keys()) == 2
        assert dist_dict['ar[init]'].shape == (10, 10)
        assert dist_dict['ar[dtw]'].shape == (10, 10)


    def test_get_behavioral_distance_ar(self):
        test_model = 'data/mock_model.p'

        model_fit = parse_model_results(test_model, resample_idx=-1,
                                        map_uuid_to_keys=True,
                                        sort_labels_by_usage=True,
                                        count='usage')

        ar_mat = model_fit['model_parameters']['ar_mat']
        assert np.array(ar_mat).shape == (100, 10, 31)

        ar_dist = get_behavioral_distance_ar(ar_mat)
        assert ar_dist.shape == (40, 40)
        assert sum(ar_dist.diagonal()) == 0

    def test_get_init_points(self):

        test_model = 'data/mock_model.p'
        model_fit = parse_model_results(test_model, resample_idx=-1,
                                        map_uuid_to_keys=True,
                                        sort_labels_by_usage=True,
                                        count='usage')

        labels = model_fit['labels']

        pca_scores = h5_to_dict('data/test_scores.h5', 'scores')

        syll_average = get_init_points(pca_scores, labels)
        assert np.array(syll_average).shape == (40, 7, 10)

    def test_reformat_dtw_distance(self):
        test_model = 'data/mock_model.p'

        model_fit = parse_model_results(test_model, resample_idx=-1,
                                        map_uuid_to_keys=True,
                                        sort_labels_by_usage=True,
                                        count='usage')

        ar_mat = model_fit['model_parameters']['ar_mat']
        assert np.array(ar_mat).shape == (100, 10, 31)

        ar_dist = get_behavioral_distance_ar(ar_mat, max_syllable=100)

        rmat = reformat_dtw_distances(ar_dist, 10, rescale=True)

        assert rmat.shape == (10, 10)