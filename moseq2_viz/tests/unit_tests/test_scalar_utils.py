import joblib
import unittest
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from unittest import TestCase
from cytoolz import merge_with
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import parse_model_results, h5_to_dict, results_to_dataframe
from moseq2_viz.scalars.util import star_valmap, remove_nans_from_labels, convert_pxs_to_mm, is_legacy, \
    generate_empty_feature_dict, convert_legacy_scalars, get_scalar_map, get_scalar_triggered_average, \
    nanzscore, _pca_matches_labels, process_scalars, find_and_load_feedback, scalars_to_dataframe, \
    make_a_heatmap, compute_all_pdf_data, compute_session_centroid_speeds, compute_mean_syll_speed


class TestScalarUtils(TestCase):

    def test_star_valmap(self):
        model_fit = 'data/mock_model.p'
        index_file = 'data/test_index_crowd.yaml'

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'

        model_data = parse_model_results(joblib.load(model_fit))
        lbl_dict = {}
        labels = model_data['labels']
        for k, v in zip(model_data['keys'], labels):
            lbl_dict[k] = np.asarray(v)

        scores_idx = h5_to_dict(index_data['pca_path'], 'scores_idx')

        def merger(d):
            return merge_with(tuple, scores_idx, d)

        out = star_valmap(remove_nans_from_labels, merger(lbl_dict))

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

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
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

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
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
        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
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

    def test_find_and_load_feedback(self):
        extract_path = 'data/proc/'
        input_path = 'data/'
        feedback_ts, feedback_status = find_and_load_feedback(extract_path, input_path)
        assert feedback_ts == None
        assert feedback_status == None


    def test_remove_nans_from_labels(self):
        lbls = np.array([[np.nan, np.nan, np.nan, 1,2,3,4,5],
                         [np.nan, np.nan, np.nan, 1,2,3,4,5]])



        out = remove_nans_from_labels(lbls, lbls)
        assert all(out == [1,2,3,4,5,1,2,3,4,5])

    def test_scalars_to_dataframe(self):
        index_file = 'data/test_index.yaml'
        model_file = 'data/mock_model.p'

        df_cols = ['angle', 'area_mm', 'area_px', 'centroid_x_mm', 'centroid_x_px',
       'centroid_y_mm', 'centroid_y_px', 'height_ave_mm', 'length_mm',
       'length_px', 'velocity_2d_mm', 'velocity_2d_px', 'velocity_3d_mm',
       'velocity_3d_px', 'velocity_theta', 'width_mm', 'width_px',
       'SessionName', 'SubjectName', 'StartTime', 'group', 'uuid']

        total_frames = 1800

        with open(index_file, 'r') as f:
            index_data = yaml.safe_load(f)
            index_data['pca_path'] = 'data/test_scores.h5'
            for i, f in enumerate(index_data['files']):
                index_data['files'][i]['path'][0] = 'data/proc/results_00.h5'
                index_data['files'][i]['path'][1] = 'data/proc/results_00.yaml'

        scalar_df = scalars_to_dataframe(index_data)

        assert isinstance(scalar_df, pd.DataFrame)
        assert all(scalar_df.columns == df_cols)
        assert scalar_df.shape == (total_frames, len(df_cols))

        # feedback timestamps
        self.assertRaises(ValueError, scalars_to_dataframe, index_data, include_model=model_file, include_feedback=True)

    def test_make_a_heatmap(self):

        index_file = 'data/test_index.yaml'

        index, sorted_index = parse_index(index_file)
        scalar_df = scalars_to_dataframe(sorted_index)

        sess = list(set(scalar_df.uuid))[0]
        centroid_vars = ['centroid_x_mm', 'centroid_y_mm']

        position = scalar_df[scalar_df['uuid'] == sess][centroid_vars].dropna(how='all').to_numpy()

        test_pdf = make_a_heatmap(position)
        assert test_pdf.shape == (50, 50)
        assert test_pdf.all() != 0

    def test_compute_all_pdf_data(self):

        index_file = 'data/test_index.yaml'

        index, sorted_index = parse_index(index_file)
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

        sessions = list(set(scalar_df.uuid))
        groups = list(set(scalar_df.group))

        scalar_df['centroid_speed_mm'] = compute_session_centroid_speeds(scalar_df)

        complete_df = compute_mean_syll_speed(complete_df, scalar_df, label_df, sessions, groups, max_sylls=40)

        assert 'speed' in complete_df.columns
        assert not complete_df.speed.isnull().all()

if __name__ == '__main__':
    unittest.main()