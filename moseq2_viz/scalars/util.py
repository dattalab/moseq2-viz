import h5py
import os
import pandas as pd
import numpy as np
import warnings
from tqdm.auto import tqdm
from moseq2_viz.util import h5_to_dict, strided_app, load_timestamps, read_yaml
from moseq2_viz.model.util import parse_model_results, _get_transitions, relabel_by_usage


# http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
# http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
# http://smeenk.com/kinect-field-of-view-comparison/
def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    """Converts x, y coordinates in pixel space to mm
    """
    cx = resolution[0] // 2
    cy = resolution[1] // 2

    xhat = coords[:, 0] - cx
    yhat = coords[:, 1] - cy

    fw = resolution[0] / (2 * np.deg2rad(field_of_view[0] / 2))
    fh = resolution[1] / (2 * np.deg2rad(field_of_view[1] / 2))

    new_coords = np.zeros_like(coords)
    new_coords[:, 0] = true_depth * xhat / fw
    new_coords[:, 1] = true_depth * yhat / fh

    return new_coords


def convert_legacy_scalars(old_features, force=False, true_depth=673.1):
    """Converts scalars in the legacy format to the new format, with explicit units.
    Args:
        old_features (str, h5 group, or dictionary of scalars): filename, h5 group, or dictionary of scalar values
        true_depth (float):  true depth of the floor relative to the camera (673.1 mm by default)

    Returns:
        features (dict): dictionary of scalar values
    """

    if type(old_features) is h5py.Group and 'centroid_x' in old_features.keys():
        print('Loading scalars from h5 dataset')
        feature_dict = {}
        for k, v in old_features.items():
            feature_dict[k] = v[...]

        old_features = feature_dict

    if (type(old_features) is str or type(old_features) is np.str_) and os.path.exists(old_features):
        print('Loading scalars from file')
        with h5py.File(old_features, 'r') as f:
            feature_dict = {}
            for k, v in f['scalars'].items():
                feature_dict[k] = v[...]

        old_features = feature_dict

    if 'centroid_x_mm' in old_features.keys() and force:
        centroid = np.hstack((old_features['centroid_x_px'][:, None],
                              old_features['centroid_y_px'][:, None]))
        nframes = len(old_features['centroid_x_mm'])

    elif not force:
        print('Features already converted')
        return None
    else:
        centroid = np.hstack((old_features['centroid_x'][:, None],
                              old_features['centroid_y'][:, None]))
        nframes = len(old_features['centroid_x'])

    features = {
        'centroid_x_px': np.zeros((nframes,), 'float32'),
        'centroid_y_px': np.zeros((nframes,), 'float32'),
        'velocity_2d_px': np.zeros((nframes,), 'float32'),
        'velocity_3d_px': np.zeros((nframes,), 'float32'),
        'width_px': np.zeros((nframes,), 'float32'),
        'length_px': np.zeros((nframes,), 'float32'),
        'area_px': np.zeros((nframes,)),
        'centroid_x_mm': np.zeros((nframes,), 'float32'),
        'centroid_y_mm': np.zeros((nframes,), 'float32'),
        'velocity_2d_mm': np.zeros((nframes,), 'float32'),
        'velocity_3d_mm': np.zeros((nframes,), 'float32'),
        'width_mm': np.zeros((nframes,), 'float32'),
        'length_mm': np.zeros((nframes,), 'float32'),
        'area_mm': np.zeros((nframes,)),
        'height_ave_mm': np.zeros((nframes,), 'float32'),
        'angle': np.zeros((nframes,), 'float32'),
        'velocity_theta': np.zeros((nframes,)),
    }

    centroid_mm = convert_pxs_to_mm(centroid, true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(centroid + 1, true_depth=true_depth)

    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)

    features['centroid_x_px'] = centroid[:, 0]
    features['centroid_y_px'] = centroid[:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion

    if 'width_px' in old_features.keys():
        features['width_px'] = old_features['width_px']
    else:
        features['width_px'] = old_features['width']

    if 'length_px' in old_features.keys():
        features['length_px'] = old_features['length_px']
    else:
        features['length_px'] = old_features['length']

    if 'area_px' in old_features.keys():
        features['area_px'] = old_features['area_px']
    else:
        features['area_px'] = old_features['area']

    if 'height_ave_mm' in old_features.keys():
        features['height_ave_mm'] = old_features['height_ave_mm']
    else:
        features['height_ave_mm'] = old_features['height_ave']

    features['width_mm'] = features['width_px'] * px_to_mm[:, 1]
    features['length_mm'] = features['length_px'] * px_to_mm[:, 0]
    features['area_mm'] = features['area_px'] * px_to_mm.mean(axis=1)

    features['angle'] = old_features['angle']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        vel_x = np.diff(np.concatenate((features['centroid_x_px'][:1], features['centroid_x_px'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_px'][:1], features['centroid_y_px'])))
        vel_z = np.diff(np.concatenate((features['height_ave_mm'][:1], features['height_ave_mm'])))

        features['velocity_2d_px'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_px'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        vel_x = np.diff(np.concatenate((features['centroid_x_mm'][:1], features['centroid_x_mm'])))
        vel_y = np.diff(np.concatenate((features['centroid_y_mm'][:1], features['centroid_y_mm'])))

        features['velocity_2d_mm'] = np.hypot(vel_x, vel_y)
        features['velocity_3d_mm'] = np.sqrt(
            np.square(vel_x)+np.square(vel_y)+np.square(vel_z))

        features['velocity_theta'] = np.arctan2(vel_y, vel_x)

    return features


def get_scalar_map(index, fill_nans=True, force_conversion=True):

    scalar_map = {}
    score_idx = h5_to_dict(index['pca_path'], 'scores_idx')

    for uuid, v in index['files'].items():

        scalars = h5_to_dict(v['path'][0], 'scalars')
        conv_scalars = convert_legacy_scalars(scalars, force=force_conversion)

        if conv_scalars is not None:
            scalars = conv_scalars

        idx = score_idx[uuid]
        scalar_map[uuid] = {}

        for k, v_scl in scalars.items():
            if fill_nans:
                scalar_map[uuid][k] = np.zeros((len(idx), ), dtype='float32')
                scalar_map[uuid][k][:] = np.nan
                scalar_map[uuid][k][~np.isnan(idx)] = v_scl
            else:
                scalar_map[uuid][k] = v_scl

    return scalar_map


def get_scalar_triggered_average(scalar_map, model_labels, max_syllable=40, nlags=20,
                                 include_keys=['velocity_2d_mm', 'velocity_3d_mm', 'width_mm',
                                             'length_mm', 'height_ave_mm', 'angle'],
                                 zscore=False):

    win = int(nlags * 2 + 1)

    # cumulative average of PCs for nlags

    if np.mod(win, 2) == 0:
        win = win + 1

    # cumulative average of PCs for nlags
    # grab the windows where 0=syllable onset

    syll_average = {}
    count = np.zeros((max_syllable, ), dtype='int')

    for scalar in include_keys:
        syll_average[scalar] = np.zeros((max_syllable, win), dtype='float32')

    for k, v in scalar_map.items():

        labels = model_labels[k]
        seq_array, locs = _get_transitions(labels)

        for i in range(max_syllable):
            hits = locs[np.where(seq_array == i)[0]]

            if len(hits) < 1:
                continue

            count[i] += len(hits)

            for scalar in include_keys:
                if scalar is 'angle':
                    use_scalar = np.diff(v[scalar])
                    use_scalar = np.insert(use_scalar, 0, 0)

                if zscore:
                    use_scalar = (v[scalar] - np.nanmean(v[scalar]))  / np.nanstd(v[scalar])
                else:
                    use_scalar = v[scalar]
                padded_scores = np.pad(use_scalar, (win // 2, win // 2),
                                   'constant', constant_values = np.nan)
                win_scores = strided_app(padded_scores, win, 1)
                syll_average[scalar][i] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        for scalar in include_keys:
            syll_average[scalar][i] /= count[i]

    return syll_average


def scalars_to_dataframe(index, include_keys=['SessionName', 'SubjectName', 'StartTime'],
                         include_model=None, disable_output=False,
                         include_pcs=False, npcs=10, include_feedback=None,
                         force_conversion=True):

    #TODO add pcs
    uuids = list(index['files'].keys())
    dset = h5_to_dict(h5py.File(index['files'][uuids[0]]['path'][0], 'r'), 'scalars')
    tmp = convert_legacy_scalars(dset, force=force_conversion)

    if tmp is not None:
        dset = tmp

    scalar_names = list(dset.keys())

    include_labels = False
    skip = []

    if include_model is not None and os.path.exists(include_model):
        labels = {}
        mdl = parse_model_results(include_model, sort_labels_by_usage=False)

        if 'train_list' in mdl.keys():
            uuids = mdl['train_list']
        else:
            uuids = mdl['keys']

        labels_usage = relabel_by_usage(mdl['labels'], count='usage')[0]
        labels_frames = relabel_by_usage(mdl['labels'], count='frames')[0]

        labels['raw'] = {k: v for k, v in zip(uuids, mdl['labels'])}
        labels['usage'] = {k: v for k, v in zip(uuids, labels_usage)}
        labels['frames'] = {k: v for k, v in zip(uuids, labels_frames)}

        label_idx = h5_to_dict(index['pca_path'], 'scores_idx')
        for uuid, lbl in labels['raw'].items():
            if len(label_idx[uuid]) != len(lbl):
                skip.append(uuid)
                continue
            labels['raw'][uuid] = lbl[~np.isnan(label_idx[uuid])]
            labels['usage'][uuid] = labels['usage'][uuid][~np.isnan(label_idx[uuid])]
            labels['frames'][uuid] = labels['frames'][uuid][~np.isnan(label_idx[uuid])]

        include_labels = True

    if include_pcs and not os.path.exists(index["pca_path"]):
        warnings.warn("PCA scores not found at {}".format(index["pca_path"]))
        include_pcs = False

    dfs = []
    for k, v in tqdm(index['files'].items(), disable=disable_output):
        if k in skip:
            continue

        _df = pd.DataFrame()

        with h5py.File(v['path'][0], 'r') as h5:
            dset = h5_to_dict(h5, 'scalars')
            if 'timestamps' in h5:
                # h5 format as of v0.1.3
                timestamps = h5['/timestamps'][()]
            elif 'timestamps' in h5['/metadata']:
                # h5 format prior to v0.1.3
                timestamps = h5['/metadata/timestamps'][()]
            else:
                raise RunTimeError("Could not find timestamps")

        dct = read_yaml(v['path'][1])
        parameters = dct['parameters']

        if include_feedback:
            if 'feedback_timestamps' in dct.keys():
                ts_data = np.array(dct['feedback_timestamps'])
                feedback_ts, feedback_status = ts_data[:, 0], ts_data[:, 1]
            else:
                feedback_path = os.path.join(os.path.dirname(parameters['input_file']),
                                             'feedback_ts.txt')
                if not os.path.exists(feedback_path):
                    feedback_path = os.path.join(os.path.dirname(v['path'][0]),
                                                 '..', 'feedback_ts.txt')

                if os.path.exists(feedback_path):
                    feedback_ts = load_timestamps(feedback_path, 0)
                    feedback_status = load_timestamps(feedback_path, 1)
                else:
                    warnings.warn('Could not find feedback file for {}'.format(v['path'][0]))
                    feedback_ts = None
                    #continue

        tmp = convert_legacy_scalars(dset, force=force_conversion)

        if tmp is not None:
            dset = tmp

        nframes = len(dset[scalar_names[0]])
        if len(timestamps) != nframes:
            warnings.warn('Timestamps not equal to number of frames for {}'.format(v['path'][0]))
            continue

        # timestamps are the index
        _df["timestamp"] = timestamps.astype('int32')
        # _df.set_index("timestamp", inplace=True)

        for scalar in scalar_names:
            _df[scalar] = dset[scalar]

        for key in include_keys:
            _df[key] = v["metadata"][key]

        _df["group"] = v["group"]
        _df["uuid"] = k

        if include_feedback and feedback_ts is not None:
            _df["feedback_status"] = -1
            _df.loc[feedback_ts.astype('int32'), "feedback_status"] = feedback_status

        if include_labels:
            _df["model_label"] = labels["raw"][k]
            _df["model_label (sort=usage)"] = labels["usage"][k]
            _df["model_label (sort=frames)"] = labels["frames"][k]
        else:
            _df["model_label"] = np.nan

        if include_pcs:
            with h5py.File(index["pca_path"], "r") as f:
                use_pcs = f["/scores/{}".format(k)][~np.isnan(label_idx[k]), :npcs]
            for _pc in range(npcs):
                _df["pc{:02d}".format(_pc)] = use_pcs[:, _pc]

        dfs.append(_df)

    scalar_df = pd.concat(dfs, axis=0)
    return scalar_df
