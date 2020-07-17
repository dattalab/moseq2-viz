import os
import h5py
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import starmap
from multiprocessing import Pool
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from cytoolz import keyfilter, itemfilter, merge_with, curry, valmap, get
from moseq2_viz.util import (h5_to_dict, strided_app, load_timestamps, read_yaml,
                             h5_filepath_from_sorted, get_timestamps_from_h5)
from moseq2_viz.model.util import parse_model_results, _get_transitions, relabel_by_usage


def _star_itemmap(func, d):
    return dict(starmap(func, d.items()))


def star_valmap(func, d):
    keys = list(d.keys())
    return dict(zip(keys, starmap(func, d.values())))

def convert_pxs_to_mm(coords, resolution=(512, 424), field_of_view=(70.6, 60), true_depth=673.1):
    '''
    Converts x, y coordinates in pixel space to mm
    # http://stackoverflow.com/questions/17832238/kinect-intrinsic-parameters-from-field-of-view/18199938#18199938
    # http://www.imaginativeuniversal.com/blog/post/2014/03/05/quick-reference-kinect-1-vs-kinect-2.aspx
    # http://smeenk.com/kinect-field-of-view-comparison/

    Parameters
    ----------
    coords (list): list of [x,y] pixel coordinate lists.
    resolution (tuple): video frame size.
    field_of_view (tuple): camera focal lengths.
    true_depth (float): detected distance between depth camera and bucket floor.

    Returns
    -------
    new_coords (list): list of same [x,y] coordinates in millimeters.
    '''

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


def is_legacy(features: dict):
    '''
    Checks a dictionary of features to see if they correspond with an older version
    of moseq.

    Parameters
    ----------
    features

    Returns
    -------
    (bool): true if the dict is from an old dataset
    '''

    old_features = ('centroid_x', 'centroid_y', 'width', 'length', 'area', 'height_ave')
    return any(x in old_features for x in features)


def generate_empty_feature_dict(nframes) -> dict:
    '''
    Generates a dict of numpy array of zeros of
    length nframes for each feature parameter.

    Parameters
    ----------
    nframes (int): length of video

    Returns
    -------
    (dict): dictionary feature to numpy 0 arrays of length nframes key-value pairs.
    '''

    features = (
        'centroid_x_px', 'centroid_y_px', 'velocity_2d_px', 'velocity_3d_px',
        'width_px', 'length_px', 'area_px', 'centroid_x_mm', 'centroid_y_mm',
        'velocity_2d_mm', 'velocity_3d_mm', 'width_mm', 'length_mm', 'area_mm',
        'height_ave_mm', 'angle', 'velocity_theta'
    )

    def make_empy_arr():
        return np.zeros((abs(nframes),), dtype='float32')
    return {k: make_empy_arr() for k in features}


def convert_legacy_scalars(old_features, force: bool = False, true_depth: float = 673.1) -> dict:
    '''
    Converts scalars in the legacy format to the new format, with explicit units.

    Parameters
    ----------
    old_features (str, h5 group, or dictionary of scalars): filename, h5 group,
    or dictionary of scalar values.
    force (bool): force the conversion of centroid_[xy]_px into mm.
    true_depth (float): true depth of the floor relative to the camera (673.1 mm by default)

    Returns
    -------
    features (dict): dictionary of scalar values
    '''

    if isinstance(old_features, h5py.Group) and 'centroid_x' in old_features:
        print('Loading scalars from h5 dataset')
        old_features = h5_to_dict(old_features, '/')

    elif isinstance(old_features, (str, np.str_)) and os.path.exists(old_features):
        print('Loading scalars from file')
        old_features = h5_to_dict(old_features, 'scalars')

    if 'centroid_x_mm' in old_features and force:
        centroid = np.hstack((old_features['centroid_x_px'][:, None],
                              old_features['centroid_y_px'][:, None]))
        nframes = len(old_features['centroid_x_mm'])
    elif not force:
        print('Features already converted')
        return old_features
    else:
        centroid = np.hstack((old_features['centroid_x'][:, None],
                              old_features['centroid_y'][:, None]))
        nframes = len(old_features['centroid_x'])

    features = generate_empty_feature_dict(nframes)

    centroid_mm = convert_pxs_to_mm(centroid, true_depth=true_depth)
    centroid_mm_shift = convert_pxs_to_mm(centroid + 1, true_depth=true_depth)

    px_to_mm = np.abs(centroid_mm_shift - centroid_mm)

    features['centroid_x_px'] = centroid[:, 0]
    features['centroid_y_px'] = centroid[:, 1]

    features['centroid_x_mm'] = centroid_mm[:, 0]
    features['centroid_y_mm'] = centroid_mm[:, 1]

    # based on the centroid of the mouse, get the mm_to_px conversion
    copy_keys = ('width', 'length', 'area')
    for key in copy_keys:
        # first try to grab _px key, then default to old version name
        features[f'{key}_px'] = get(f'{key}_px', old_features, old_features[key])

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


def get_scalar_map(index, fill_nans=True, force_conversion=False):
    '''
    Returns a dictionary of scalar values loaded from an index dictionary.

    Parameters
    ----------
    index (dict): dictionary of index file contents.
    fill_nans (bool): indicate whether to replace NaN values with 0.
    force_conversion (bool): force the conversion of centroid_[xy]_px into mm.

    Returns
    -------
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    '''

    scalar_map = {}
    score_idx = h5_to_dict(index['pca_path'], 'scores_idx')

    try:
        iter_items = index['files'].items()
    except:
        iter_items = enumerate(index['files'])

    for i, v in iter_items:
        if isinstance(index['files'], list):
            uuid = index['files'][i]['uuid']
        elif isinstance(index['files'], dict):
            uuid = i

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
    '''
    Get averages of selected scalar keys for each syllable.

    Parameters
    ----------
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    model_labels (dict): dictionary of uuid to syllable label array pairs.
    max_syllable (int): maximum number of syllables to use.
    nlags (int): number of lags to use when averaging over a series of PCs.
    include_keys (list): list of scalar values to load averages of.
    zscore (bool): indicate whether to z-score loaded values.

    Returns
    -------
    syll_average (dict): dictionary of scalars for each syllable sequence.
    '''

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
                use_scalar = v[scalar]
                if scalar == 'angle':
                    use_scalar = np.diff(use_scalar)
                    use_scalar = np.insert(use_scalar, 0, 0)
                if zscore:
                    use_scalar = nanzscore(use_scalar)

                padded_scores = np.pad(use_scalar, (win // 2, win // 2),
                                       'constant', constant_values=np.nan)
                win_scores = strided_app(padded_scores, win, 1)
                syll_average[scalar][i] += np.nansum(win_scores[hits, :], axis=0)

    for i in range(max_syllable):
        for scalar in include_keys:
            syll_average[scalar][i] /= count[i]

    return syll_average


def nanzscore(data):
    '''
    Z-score numpy array that may contain NaN values.

    Parameters
    ----------
    data (np.ndarray): array of scalar values.

    Returns
    -------
    data (np.ndarray): z-scored data.
    '''

    return (data - np.nanmean(data)) / np.nanstd(data)


def _pca_matches_labels(pca, labels):
    '''
    Make sure that the number of frames in the pca dataset matches the
    number of frames in the assigned labels.

    Parameters
    ----------
    pca (np.array): array of session PC scores.
    labels (np.array): array of session syllable labels

    Returns
    -------
    (bool): indicates whether the PC scores length matches the corresponding assigned labels.
    '''

    return len(pca) == len(labels)


def process_scalars(scalar_map: dict, include_keys: list, zscore: bool = False) -> dict:
    '''
    Fill NaNs and possibly zscore scalar values.

    Parameters
    ----------
    scalar_map (dict): dictionary of all the scalar values acquired after extraction.
    include_keys (list): scalar keys to process.
    zscore (bool): indicate whether to z-score loaded values.

    Returns
    -------

    '''

    out = defaultdict(list)
    for k, v in scalar_map.items():
        for scalar in include_keys:
            use_scalar = v[scalar]
            if scalar == 'angle':
                use_scalar = np.diff(use_scalar)
                use_scalar = np.insert(use_scalar, 0, 0)
            if zscore:
                use_scalar = nanzscore(use_scalar)
            out[k].append(use_scalar)
    return valmap(np.array, out)


def find_and_load_feedback(extract_path, input_path):
    join = os.path.join
    feedback_path = join(os.path.dirname(input_path), 'feedback_ts.txt')
    if not os.path.exists(feedback_path):
        feedback_path = join(os.path.dirname(extract_path), '..', 'feedback_ts.txt')

    if os.path.exists(feedback_path):
        feedback_ts = load_timestamps(feedback_path, 0)
        feedback_status = load_timestamps(feedback_path, 1)
        return feedback_ts, feedback_status
    else:
        warnings.warn(f'Could not find feedback file for {extract_path}')
        return None, None


def remove_nans_from_labels(idx, labels):
    '''
    Removes the frames from `labels` where `idx` has NaNs in it.

    Parameters
    ----------
    idx (list): indices to remove NaN values at.
    labels (list): label list containing NaN values.

    Returns
    -------
    (list): label list excluding NaN values at given indices
    '''

    return labels[~np.isnan(idx)]


def scalars_to_dataframe(index: dict, include_keys: list = ['SessionName', 'SubjectName', 'StartTime'],
                         include_model=None, disable_output=False, include_pcs=False, npcs=10,
                         include_feedback=None, force_conversion=True, include_labels=False):
    '''
    Generates a dataframe containing scalar values over the course of a recording session.
    If a model string is included, then return only animals that were included in the model
    Called to sort scalar metadata information when graphing in plot-scalar-summary.

    Parameters
    ----------
    index (dict): a sorted_index generated by `parse_index` or `get_sorted_index`
    include_keys (list): a list of other moseq related keys to include in the dataframe
    include_model (str): path to an existing moseq model
    disable_output (bool): indicate whether to show tqdm output.
    include_pcs (bool): UNUSED
    npcs (int): UNUSED
    include_feedback (bool): indicate whether to include timestamp data
    force_conversion (bool): force the conversion of centroid_[xy]_px into mm.
    include_labels (bool): UNUSED

    Returns
    -------
    scalar_df (pandas DataFrame): DataFrame of loaded scalar values with their selected metadata.
    '''

    scalar_dict = defaultdict(list)
    skip = []

    files = index['files']
    try:
        uuids = list(files.keys())
        # use dset from first animal to generate a list of scalars
        dset = h5_to_dict(h5_filepath_from_sorted(files[uuids[0]]), path='scalars')
    except:
        uuids = [f['uuid'] for f in index['files']]
        # use dset from first animal to generate a list of scalars
        dset = h5_to_dict(h5_filepath_from_sorted(files[0]), path='scalars')

    # only convert if the dataset is legacy and conversion is forced
    if is_legacy(dset) and force_conversion:
        dset = convert_legacy_scalars(dset, force=force_conversion)

    # generate a list of scalars
    scalar_names = list(dset.keys())

    # get model labels for UUIDs if a model was supplied
    if include_model and os.path.exists(include_model):
        mdl = parse_model_results(include_model, sort_labels_by_usage=False, map_uuid_to_keys=True)

        labels = mdl['labels']

        if isinstance(labels, dict):
            keys = list(labels.keys())
            labels = np.array(list(labels.values()))

        # we need to call these functions here so we don't filter out data before relabelling
        usage, _ = relabel_by_usage(labels, count='usage')
        frames, _ = relabel_by_usage(labels, count='frames')

        if isinstance(usage, list) or isinstance(usage, np.ndarray):
            usage = {k:v for k,v in zip(keys, usage)}
            frames = {k: v for k, v in zip(keys, frames)}
            labels = {k: v for k, v in zip(keys, labels)}
            files = {k: v for k, v in zip(keys, files)}

        # get pca score indices
        scores_idx = h5_to_dict(index['pca_path'], 'scores_idx')
        # only include labels with the same number of frames as PCA
        labels = itemfilter(lambda a: _pca_matches_labels(scores_idx[a[0]], a[1]), mdl['labels'])

        labelfilter = curry(keyfilter)(lambda k: k in labels)

        # filter the relabeled dictionaries too
        usage = labelfilter(usage)
        frames = labelfilter(frames)
        # filter the pca scores to match labels
        scores_idx = labelfilter(scores_idx)

        # make function to merge syllables with the syll scores
        def merger(d):
            return merge_with(tuple, scores_idx, d)

        # use the scores to remove the nans
        usage = star_valmap(remove_nans_from_labels, merger(usage))
        frames = star_valmap(remove_nans_from_labels, merger(frames))
        labels = star_valmap(remove_nans_from_labels, merger(labels))

        # only include extractions that were modeled and fit the above criteria
        files = keyfilter(lambda x: x in labels, files)

    try:
        iter_items = files.items()
    except:
        iter_items = enumerate(files)

    for k, v in tqdm(iter_items, disable=disable_output):

        pth = h5_filepath_from_sorted(v)
        dset = h5_to_dict(pth, 'scalars')

        # get extraction parameters for this h5 file
        dct = read_yaml(v['path'][1])
        parameters = dct['parameters']

        if include_feedback:
            if 'feedback_timestamps' in dct:
                ts_data = np.array(dct['feedback_timestamps'])
                feedback_ts, feedback_status = ts_data[:, 0], ts_data[:, 1]
            else:
                feedback_ts, feedback_status = find_and_load_feedback(pth, parameters['input_file'])

        # convert scalar names into modern format if they are legacy
        if is_legacy(dset) and force_conversion:
            dset = convert_legacy_scalars(dset, force=force_conversion)

        nframes = len(dset[scalar_names[0]])
        if include_feedback:
            try:
                timestamps = get_timestamps_from_h5(pth)
                scalar_dict['timestamp'] = timestamps.astype('int32')
            except: #h5 file path exception, maybe Attribute or KeyError
                warnings.warn(f'timestamps for {pth} were not found')
                warnings.warn('This could be due to a missing/incorrectly named timestamp file in that session directory.')
                warnings.warn('If the file does exist, ensure it has the correct name/location and re-extract the session.')
                pass
            if len(timestamps) != nframes:
                warnings.warn(f'Timestamps not equal to number of frames for {pth}, skipping')
                continue

        # add scalar data for this animal
        for scalar in scalar_names:
            scalar_dict[scalar] += dset[scalar].tolist()

        # add metadata from `include_keys`
        for key in include_keys:
            # every frame should have the same amount of metadata
            scalar_dict[key] += [v['metadata'][key]] * nframes

        scalar_dict['group'] += [v['group']] * nframes
        scalar_dict['uuid'] += [k] * nframes

        if include_feedback and feedback_ts is not None:
            for ts in timestamps:
                hit = np.where(ts.astype('int32') == feedback_ts.astype('int32'))[0]
                if len(hit) > 0:
                    scalar_dict['feedback_status'] += [feedback_status[hit]]
                else:
                    scalar_dict['feedback_status'] += [-1]
        elif include_feedback:
            scalar_dict['feedback_status'] += [-1] * nframes

        if include_model and os.path.exists(include_model):
            scalar_dict['model_label'] += labels[k].tolist()
            scalar_dict['model_label (sort=usage)'] += usage[k].tolist()
            scalar_dict['model_label (sort=frames)'] += frames[k].tolist()

    # turn each key in scalar_names into a numpy array
    for scalar in scalar_names:
        scalar_dict[scalar] = np.array(scalar_dict[scalar])

    # return scalar_dict
    scalar_df = pd.DataFrame(scalar_dict)

    return scalar_df


def make_a_heatmap(position):
    '''
    Uses a kernel density function to create a heatmap representing the mouse position throughout a single session.

    Parameters
    ----------
    position (2d numpy array): 2d array of mouse centroid coordinates (for a single session),
     computed from compute_session_centroid_speeds.


    Returns
    -------
    pdf (2d numpy array): shape (50, 50) representing the PDF for the mouse position over the whole session.

    '''

    n_grid = 50

    # Set up the bounds over which to build the KDE
    X, Y = np.meshgrid(* \
                           [np.linspace(*np.percentile(d, [0.01, 99.99]), num=n_grid)
                            for d in (position[:, 0], position[:, 1])
                            ])
    position_grid = np.hstack((X.ravel()[:, None], Y.ravel()[:, None]))
    bandwidth = (X.max() - X.min()) / 25.0

    # Set up the KDE
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(position)
    pdf = np.exp(kde.score_samples(position_grid)).reshape(n_grid, n_grid)
    return pdf


def compute_all_pdf_data(scalar_df, normalize=False, centroid_vars=['centroid_x_mm', 'centroid_y_mm']):
    '''
    Computes a position PDF for all sessions and returns the pdfs with corresponding lists of
     groups, session uuids, and subjectNames.

    Parameters
    ----------
    scalar_df (pd.DataFrame): DataFrame containing all scalar data + uuid columns for all stacked sessions
    normalize (bool): Indicates whether normalize the pdfs.
    centroid_vars (list): list of strings for column values to use when computing mouse position.

    Returns
    -------
    pdfs (list): list of 2d np.arrays of PDFs for each session.
    groups (list): list of strings of groups corresponding to pdfs index.
    sessions (list): list of strings of session uuids corresponding to pdfs index.
    subjectNames (list): list of strings of subjectNames corresponding to pdfs index.
    '''

    sessions = list(set(scalar_df.uuid))
    groups, positions, subjectNames = [], [], []

    for sess in tqdm(sessions):
        groups.append(scalar_df[scalar_df['uuid'] == sess][['group']].iloc[0][0])
        positions.append(scalar_df[scalar_df['uuid'] == sess][centroid_vars].dropna(how='all').to_numpy())
        subjectNames.append(scalar_df[scalar_df['uuid'] == sess][['SubjectName']].iloc[0][0])

    pool_ = Pool()
    pdfs = pool_.map(make_a_heatmap, np.array(positions))
    pdfs = np.stack(pdfs).copy()
    pool_.close()

    clim_ = 0, pdfs[0].max()

    if normalize:
        return np.stack([p / p.sum() for p in pdfs]), groups, sessions, subjectNames

    return pdfs, groups, sessions, subjectNames


def compute_session_centroid_speeds(scalar_df, grouping_keys=['uuid', 'group'],
                                    centroid_keys=['centroid_x_mm', 'centroid_y_mm']):
    '''
    Computes the centroid speed float value of the mouse given the Series of  mm x and y coordinates
     from the scalar_df DataFrame.

    Parameters
    ----------
    scalar_df (pd.DataFrame): DataFrame containing all scalar data + uuid columns for all stacked sessions
    grouping_keys (list): list of column names to group the df keys by
    centroid_keys (list): list of column names containing the centroid values.

    Returns
    -------
    sc_speed (pd.DataFrame): single column of a DataFrame containing centroid value to be appended
    as new column to scalar_df

    '''

    use_df = scalar_df[centroid_keys + grouping_keys]
    sc_speed = (use_df.centroid_x_mm * 2.0).diff() ** 2 + (use_df.centroid_y_mm * 2.0).diff() ** 2

    return sc_speed

def compute_mean_syll_speed(complete_df, scalar_df, label_df, groups=None, max_sylls=40):
    '''
    Computes the mean syllable speed based on the centroid speed of the mouse at the frame indices
     with corresponding label values.

    Parameters
    ----------
    complete_df (pd.DataFrame): DataFrame containing syllable statistic results for each uuid.
    scalar_df (pd.DataFrame): DataFrame containing all scalar data + uuid columns for all stacked sessions
    label_df (pd.DataFrame): DataFrame containing syllable labels at each frame (nsessions rows x max(nframes) cols)
    sessions (list): list of strings of session uuids corresponding to pdfs index.
    groups (list): list of strings of groups corresponding to pdfs index.
    max_sylls (int): maximum amount of syllables to include in output.

    Returns
    -------
    complete_df (pd.DataFrame): updated input dataframe with a speed value for each syllable merge in as a new column.
    '''

    lbl_df = label_df.T
    columns = lbl_df.columns
    gk = ['group', 'uuid']

    centroid_speeds = scalar_df[['centroid_speed_mm'] + gk]

    if isinstance(groups, list) or isinstance(groups, tuple):
        if len(groups) == 0:
            groups = None

    all_sessions = []

    for col in tqdm(columns, total=len(columns), desc='Computing Per Session Syll Speeds'):
        if groups != None:
            if col[0] not in groups:
                continue

        sess_lbls = lbl_df[col].iloc[3:].reset_index().dropna(axis=0, how='all')
        sess_speeds = centroid_speeds[centroid_speeds['uuid'] == col[1]].iloc[3:].reset_index()

        sess_dict = {
            'uuid': [],
            'syllable': [],
            'speed': []
        }
        for lbl in range(max_sylls):
            indices = (sess_lbls[col] == lbl)

            mean_lbl_speed = np.nanmean(sess_speeds[indices].centroid_speed_mm)

            sess_dict['uuid'].append(col[1])
            sess_dict['syllable'].append(lbl)
            sess_dict['speed'].append(mean_lbl_speed)

        all_sessions.append(sess_dict)

    all_speeds_df = pd.DataFrame.from_dict(all_sessions[0])
    y = all_speeds_df['speed']
    all_speeds_df['speed'] = np.where(y.between(0, 300), y, 0)

    for i in range(1, len(all_sessions)):
        tmp_df = pd.DataFrame.from_dict(all_sessions[i])
        y = tmp_df['speed']
        tmp_df['speed'] = np.where(y.between(0, 300), tmp_df['speed'], 0)

        all_speeds_df = all_speeds_df.append(tmp_df)

    complete_df = pd.merge(complete_df, all_speeds_df, on=['uuid', 'syllable'])

    return complete_df