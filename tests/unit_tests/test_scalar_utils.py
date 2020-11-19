import joblib
import unittest
import numpy as np
import pandas as pd
from unittest import TestCase
from cytoolz import merge_with
from moseq2_viz.util import parse_index, read_yaml
from moseq2_viz.model.util import parse_model_results, h5_to_dict, results_to_dataframe
from moseq2_viz.scalars.util import star_valmap, convert_pxs_to_mm, is_legacy, \
    generate_empty_feature_dict, convert_legacy_scalars, get_scalar_map, get_scalar_triggered_average, \
    nanzscore, _pca_matches_labels, process_scalars, scalars_to_dataframe, \
    compute_all_pdf_data, compute_session_centroid_speeds, compute_mean_syll_scalar, \
    compute_syllable_position_heatmaps, get_syllable_pdfs, compute_mouse_dist_to_center, \
    h5_filepath_from_sorted

class TestScalarUtils(TestCase):

    def test_star_valmap(self):
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        lbl_dict = {}
        labels = model_data['labels']
        for k, v in zip(model_data['keys'], labels):
            lbl_dict[k] = np.asarray(v)

        scores_idx = h5_to_dict(index_data['pca_path'], 'scores_idx')

        def merger(d):
            return merge_with(tuple, scores_idx, d)

        out = star_valmap(np.isnan, merger(lbl_dict))

        assert isinstance(out, dict)
        for v in out.values():
            assert all(np.isnan(v)) == False

    def test_convert_pxs_to_mm(self):
        coords = np.asarray([[0, 0], [256, 212],[512, 424]])
        resolution = (512, 424)
        field_of_view = (70.6, 60)
        true_depth = 673.1

        cx = resolution[0] // 2
        cy = resolution[1] // 2

        xhat = coords[:, 0] - cx
        yhat = coords[:, 1] - cy

        fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
        fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

        new_coords = np.zeros_like(coords)
        new_coords[:, 0] = true_depth * xhat / fw
        new_coords[:, 1] = true_depth * yhat / fh

        test_coords = convert_pxs_to_mm(coords, resolution, field_of_view, true_depth)
        np.testing.assert_array_equal(new_coords, test_coords)
        assert np.isnan(test_coords).all() == False

    def test_is_legacy(self):
        true_test = {'centroid_x': [1], 'area': 2}
        false_test = {'centroid_x_mm': [1], 'area_mm': 2}

        assert is_legacy(true_test) == True
        assert is_legacy(false_test) == False

    def test_generate_empty_feature_dict(self):

        num_frames = 900
        empty_dict = generate_empty_feature_dict(num_frames)

        for v in empty_dict.values():
            np.testing.assert_array_equal(v, np.zeros((abs(num_frames),), dtype='float32'))

    def test_convert_legacy_scalars(self):
        test_file = 'data/proc/results_00.h5'
        out_feat = convert_legacy_scalars(test_file)
        assert isinstance(out_feat, dict)
        assert is_legacy(out_feat) == False
        self.assertRaises(KeyError, convert_legacy_scalars, test_file, force=True)

        # setting up values to convert
        out_feat['height_ave_mm'] = [1]
        out_feat['width'] = [1]
        out_feat['width_mm'] = [1]
        out_feat['width_px'] = [1]
        out_feat['length'] = [1]
        out_feat['length_mm'] = [1]
        out_feat['length_px'] = [1]
        out_feat['area'] = [1]
        out_feat['area_mm'] = [1]
        out_feat['area_px'] = [1]

        feats = convert_legacy_scalars(out_feat, force=True)

        assert feats != out_feat

    def test_get_scalar_map(self):
        index_file = 'data/test_index_crowd.yaml'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i, f in enumerate(index_data['files']):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        test_scalar_map = get_scalar_map(index_data)
        scalar_keys = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                       'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                       'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                       'velocity_3d_px', 'velocity_theta', 'width_mm', 'width_px']

        for v in test_scalar_map.values():
            assert list(v.keys()) == scalar_keys
        assert len(test_scalar_map.keys()) == 2
        assert isinstance(test_scalar_map, dict)

        self.assertRaises(KeyError, get_scalar_map, index_data, force_conversion=True)


    def test_scalar_triggered_average(self):
        index_file = 'data/test_index_crowd.yaml'
        model_fit = 'data/mock_model.p'

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i, f in enumerate(index_data['files']):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        lbl_dict = {}
        labels = model_data['labels']
        for k, v in zip(model_data['keys'], labels):
            lbl_dict[k] = np.asarray(v)

        max_syllable = 40

        test_scalar_map = get_scalar_map(index_data)
        test_syll_avg = get_scalar_triggered_average(test_scalar_map, lbl_dict,
                                                     max_syllable=max_syllable)

        avg_keys = ['velocity_2d_mm', 'velocity_3d_mm', 'width_mm',
                    'length_mm', 'height_ave_mm', 'angle']

        assert avg_keys == list(test_syll_avg.keys())
        for v in test_syll_avg.values():
            assert len(v) == max_syllable
        assert isinstance(test_syll_avg, dict)

        # primitive zscore test
        test_syll_z_avg = get_scalar_triggered_average(test_scalar_map, lbl_dict,
                                               max_syllable=max_syllable, zscore=True)

        assert test_syll_avg.values() != test_syll_z_avg.values()


    def test_nanzscore(self):
        data = np.array([[1,2,3,4,5],[np.nan,np.nan,np.nan,np.nan,np.nan],[6,7,8,9,10]])
        nanz = nanzscore(data)
        print(nanz)
        assert all(nanz[0] < 0)
        assert np.isnan(nanz[1]).all() == True
        assert all(nanz[2] > 0)

    def test_pca_matches_labels(self):
        pca = [1,2,3,4,5,6,7,8,9,10]
        labels = [0,1,2,3,4,5,6,7,8,9]

        assert _pca_matches_labels(pca,labels) == True
        labels = [0,1,2,3]
        assert _pca_matches_labels(pca,labels) == False

    def test_process_scalars(self):
        index_file = 'data/test_index_crowd.yaml'
        scalar_keys = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                       'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                       'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                       'velocity_3d_px', 'velocity_theta', 'width_mm', 'width_px']

        num_frames = 908
        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i, f in enumerate(index_data['files']):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'

        test_scalar_map = get_scalar_map(index_data)
        mapped = process_scalars(test_scalar_map, scalar_keys)

        assert isinstance(mapped, dict)
        assert len(mapped.keys()) == 2

        z_mapped = process_scalars(test_scalar_map, scalar_keys, zscore=True)
        assert isinstance(z_mapped, dict)

        for v1, v2 in zip(mapped.values(), z_mapped.values()):
            assert v1.all() != v2.all()
            assert np.asarray(v1).shape == (17, 908)
            assert np.asarray(v2).shape == (17, 908)

    def test_scalars_to_dataframe(self):
        index_file = 'data/test_index.yaml'

        df_cols = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
                   'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
                   'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
                   'velocity_3d_px', 'velocity_theta', 'width_mm', 'width_px',
                   'dist_to_center_px', 'group', 'uuid', 'h5_path', 'timestamps',
                   'frame index', 'SessionName', 'SubjectName', 'StartTime']

        total_frames = 1800

        index_data = read_yaml(index_file)
        index_data['pca_path'] = 'data/test_scores.h5'
        for i, f in enumerate(index_data['files']):
            index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
            index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        scalar_df = scalars_to_dataframe(index_data)

        assert isinstance(scalar_df, pd.DataFrame)
        assert all(scalar_df.columns == df_cols)
        assert scalar_df.shape == (total_frames, len(df_cols))

    def test_compute_all_pdf_data(self):

        index_file = 'data/test_index.yaml'

        _, sorted_index = parse_index(index_file)
        scalar_df = scalars_to_dataframe(sorted_index)

        test_pdfs, groups, sessions, subjectNames = compute_all_pdf_data(scalar_df)

        assert len(test_pdfs) == len(groups) == len(sessions) == len(subjectNames)
        for i in range(len(test_pdfs)):
            assert test_pdfs[i].shape == (50, 50)

    def test_compute_session_centroid_speeds(self):
        index_file = 'data/test_index.yaml'

        index, sorted_index = parse_index(index_file)
        scalar_df = scalars_to_dataframe(sorted_index)

        new_col = compute_session_centroid_speeds(scalar_df)

        assert str(new_col.iloc[0]) == 'nan' # no speed recorded in first frame of recording
        assert len(new_col) == len(scalar_df)

    def test_compute_mean_syll_speed(self):
        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'

        _, sorted_index = parse_index(test_index)
        scalar_df = scalars_to_dataframe(sorted_index)
        complete_df, label_df = results_to_dataframe(test_model, sorted_index, compute_labels=True)

        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)
        scalar_df['syllable'] = np.inf
        for i, indexes in enumerate(label_df.index):
            session_label_idx = scalar_df[scalar_df['uuid'] == indexes[1]].index
            scalar_df.loc[session_label_idx, 'syllable'] = list(label_df.iloc[i])[:len(session_label_idx)]

        complete_df = compute_mean_syll_scalar(complete_df, scalar_df, scalar='centroid_speed_mm', max_sylls=40)

        assert 'centroid_speed_mm' in complete_df.columns
        assert not complete_df.centroid_speed_mm.isnull().all()

    def test_compute_mouse_dist_to_center(self):

        test_index = 'data/test_index.yaml'

        _, sorted_index = parse_index(test_index)

        files = sorted_index['files']

        uuids = list(files.keys())
        dset = h5_to_dict(h5_filepath_from_sorted(files[uuids[0]]), path='scalars')

        # Get ROI shape to compute distance to center
        roi = h5_to_dict(h5_filepath_from_sorted(files[uuids[0]]), path='metadata/extraction/roi')['roi'].shape

        centroid_x_px = dset['centroid_x_px']
        centroid_y_px = dset['centroid_y_px']

        dist_to_center = compute_mouse_dist_to_center(roi, centroid_x_px, centroid_y_px)

        assert len(dist_to_center) == 900
        assert all(x < 1.2 for x in dist_to_center)

    def test_compute_syllable_position_heatmaps(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        max_sylls = 20

        _, sorted_index = parse_index(test_index)

        scalar_df = scalars_to_dataframe(sorted_index)

        model = parse_model_results(joblib.load(test_model))
        df, label_df = results_to_dataframe(model_dict=model, index_dict=sorted_index, sort=True, compute_labels=True)

        df = compute_syllable_position_heatmaps(df, scalar_df, label_df, syllables=range(max_sylls))

        assert 'pdf' in df.columns
        assert all(x.all() != np.zeros((50, 50)).all() for x in df['pdf'])
        assert len(df['pdf']) == 40

    def test_get_syllable_pdfs(self):

        test_index = 'data/test_index.yaml'
        test_model = 'data/test_model.p'
        max_sylls = 20

        _, sorted_index = parse_index(test_index)

        scalar_df = scalars_to_dataframe(sorted_index)

        model = parse_model_results(joblib.load(test_model))
        df, label_df = results_to_dataframe(model_dict=model, index_dict=sorted_index, sort=True, compute_labels=True)

        df = compute_syllable_position_heatmaps(df, scalar_df, label_df, syllables=range(max_sylls))

        gdf, groups = get_syllable_pdfs(df, normalize=True, syllables=range(max_sylls), groupby='group')

        assert len(gdf) == 1
        assert gdf[0].shape == (20, 50, 50)
        assert len(groups) == 1

if __name__ == '__main__':
    unittest.main()