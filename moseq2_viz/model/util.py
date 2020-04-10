import os
import h5py
import glob
import tqdm
import joblib
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import starmap
from numpy import linalg as LA
from cytoolz.curried import get
from sklearn.cluster import KMeans
from os.path import join, basename, dirname
from typing import Iterator, Any, Dict, Union
from collections import defaultdict, OrderedDict
from scipy.optimize import linear_sum_assignment
from moseq2_viz.util import np_cache, h5_to_dict, star
from cytoolz import curry, valmap, compose, complement, itemmap, concat

def merge_models(model_dir, ext):
    '''
    Merges model states by using the Hungarian Algorithm:
    a minimum distance state matching algorithm. User inputs a
    directory containing models to merge, (and the name of the latest-trained
    model) to match other model states to.

    Parameters
    ----------
    model_dir (str): path to directory containing all the models to merge.
    ext (str): model extension to search for.

    Returns
    -------
    model_data (dict): a dictionary containing all the new
    keys and state-matched labels.
    '''

    tmp = os.path.join(model_dir, '*.'+ext.strip('.'))
    model_paths = [m for m in glob.glob(tmp)]

    model_data = {}
    for m, model_fit in enumerate(model_paths):
        unit_data = parse_model_results(joblib.load(model_fit))
        for k,v in unit_data.items():
            if k not in list(model_data.keys()):
                model_data[k] = v
            else:
                try:
                    if k == 'model_parameters':
                        prev = model_data[k]['ar_mat']
                        curr_arrays = v['ar_mat']
                        cost = np.zeros((len(prev), len(curr_arrays)))

                        for i, state1 in enumerate(prev):
                            for j, state2 in enumerate(curr_arrays):
                                distance = LA.norm(abs(state1 - state2))
                                cost[i][j] = distance

                        row_ind, col_ind = linear_sum_assignment(cost)
                        mapping = {c:r for r,c in zip(row_ind, col_ind)}
                        adjusted_labels = []

                        for oldlbl in unit_data['labels'][0]:
                            try:
                                adjusted_labels.append(mapping[oldlbl])
                            except:
                                pass

                        model_data['labels'].append(np.array(adjusted_labels))
                    elif k == 'keys' or k == 'train_list':
                        for i in v:
                            model_data[k].append(i)
                except:
                    print('Error, trying to merge models with unequal number of PCs.')
                    pass
    return model_data

def _get_transitions(label_sequence):
    '''
    Computes labels switch to another label. Throws out the first state (usually
    labeled as -5).
    Parameters
    ----------
    label_sequence (tuple): a tuple of syllable transitions and their indices

    Returns
    -------

    '''

    arr = deepcopy(label_sequence)

    # get syllable transition locations
    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]

    return transitions, locs


def _whiten_all(pca_scores: Dict[str, np.ndarray], center=True):
    '''
    Whitens all PC scores at once.
    Parameters
    ----------
    pca_scores (dict): dictionary of uuid to PC score key-value pairs
    center (bool): decide whether to normalize data with an offset value.

    Returns
    -------
    whitened_scores (dict): whitened pca_scores dict
    '''

    valid_scores = np.concatenate([x[~np.isnan(x).any(axis=1), :] for x in pca_scores.values()])
    mu, cov = valid_scores.mean(axis=0), np.cov(valid_scores, rowvar=False, bias=1)

    L = np.linalg.cholesky(cov)

    if center:
        offset = 0
    else:
        offset = mu

    whitened_scores = deepcopy(pca_scores)

    for k, v in whitened_scores.items():
        whitened_scores[k] = np.linalg.solve(L, (v - mu).T).T + offset

    return whitened_scores


# per https://gist.github.com/tg12/d7efa579ceee4afbeaec97eb442a6b72
def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False) -> list:
    '''
    Compute the transition matrix from a set of model labels.
    Parameters
    ----------
    labels (list of np.array of ints): labels loaded from a model fit
    max_syllable (int): maximum syllable number to consider
    normalize (str): how to normalize transition matrix, 'bigram' or 'rows' or 'columns'
    smoothing (float): constant to add to transition_matrix pre-normalization to smooth counts
    combine (bool): compute a separate transition matrix for each element (False)
    or combine across all arrays in the list (True)
    disable_output (bool): verbosity

    Returns
    -------
    transition_matrix (list): list of 2d np.arrays that represent the transitions
            from syllable i (row) to syllable j (column)
    '''

    if combine:
        init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing

        for v in labels:

            transitions = _get_transitions(v)[0]

            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

        if normalize == 'bigram':
            init_matrix /= init_matrix.sum()
        elif normalize == 'rows':
            init_matrix /= init_matrix.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            init_matrix /= init_matrix.sum(axis=0, keepdims=True)
        else:
            pass

        all_mats = init_matrix
    else:

        all_mats = []
        for v in tqdm.tqdm(labels, disable=disable_output):

            init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing
            transitions = _get_transitions(v)[0]

            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

            if normalize == 'bigram':
                init_matrix /= init_matrix.sum()
            elif normalize == 'rows':
                init_matrix /= init_matrix.sum(axis=1, keepdims=True)
            elif normalize == 'columns':
                init_matrix /= init_matrix.sum(axis=0, keepdims=True)
            else:
                pass

            all_mats.append(init_matrix)

    return all_mats


def get_mouse_syllable_slices(syllable: int, labels: np.ndarray) -> Iterator[slice]:
    '''
    Return a generator containing slices of `syllable` indices for a mouse.
    Parameters
    ----------
    syllable (list): list of syllables to get slices from.
    labels (np.ndarrary): list of label predictions for each session.

    Returns
    -------
    slices (list): list of syllable label slices; e.g. [slice(3, 6, None), slice(9, 12, None)]
    '''

    labels = np.concatenate(([-1], labels, [-1]))
    is_syllable = np.diff(np.int16(labels == syllable))
    starts = np.where(is_syllable == 1)[0]
    ends = np.where(is_syllable == -1)[0]
    slices = starmap(slice, zip(starts, ends))
    return slices


any_nan = compose(np.any, np.isnan)
non_nan = complement(any_nan)


@curry
def syllable_slices_from_dict(syllable: int, labels: Dict[str, np.ndarray], index: Dict,
                              filter_nans: bool = True) -> Dict[str, list]:
    '''
    Reads dictionary of syllable labels, and returning a dict of syllable slices.
    Parameters
    ----------
    syllable (list): list of syllables to get slices from.
    labels (np.ndarrary): list of label predictions for each session.
    index (dict): index file contents contained in a dict.
    filter_nans (bool): replace NaN values with 0.

    Returns
    -------
    vals (dict): key-value pairs of syllable slices per session uuid.
    '''

    getter = curry(get_mouse_syllable_slices)(syllable)
    vals = valmap(getter, labels)

    score_idx = h5_to_dict(index['pca_path'], '/scores_idx')

    def filter_score(uuid, slices):
        mouse = score_idx[uuid]
        get_nan = compose(non_nan, get(seq=mouse))
        return uuid, filter(get_nan, slices)

    # filter out slices with NaNs
    if filter_nans:
        vals = itemmap(star(filter_score), vals)

    vals = valmap(list, vals)
    # TODO: array length mismatch warnings?
    return vals


@curry
def get_syllable_slices(syllable, labels, label_uuids, index, trim_nans: bool = True) -> list:
    '''
    Get the indices that correspond to a specific syllable for each animal in a modeling run.
    Parameters
    ----------
    syllable (list): list of syllables to get slices from.
    labels (np.ndarrary): list of label predictions for each session.
    label_uuids (list): list of uuid keys corresponding to each session.
    index (dict): index file contents contained in a dict.
    trim_nans (bool): flag to use the pca scores file for removing time points that contain NaNs.
    Only use if you have not already trimmed NaNs previously (i.e. in `scalars_to_dataframe`).

    Returns
    -------
    syllable_slices (list): a list of indices for `syllable` in the `labels` array. Each item in the list
    is a tuple of (slice, uuid, h5_file).
    '''

    try:
        h5s = [v['path'][0] for v in index['files'].values()]
        h5_uuids = list(index['files'].keys())
    except:
        h5s = [v['path'][0] for v in index['files']]
        h5_uuids = [v['uuid'] for v in index['files']]


    # grab the original indices from the pca file as well...
    if trim_nans:
        with h5py.File(index['pca_path'], 'r') as f:
            score_idx = h5_to_dict(f, 'scores_idx')

    sorted_h5s = [h5s[h5_uuids.index(uuid)] for uuid in label_uuids]
    syllable_slices = []

    for label_arr, label_uuid, h5 in zip(labels, label_uuids, sorted_h5s):

        if trim_nans:
            idx = score_idx[label_uuid]

            if len(idx) > len(label_arr):
                warnings.warn('Index length {:d} and label array length {:d} in {}'
                              .format(len(idx), len(label_arr), h5))
                idx = idx[:len(label_arr)]
            elif len(idx) < len(label_arr):
                warnings.warn('Index length {:d} and label array length {:d} in {}'
                              .format(len(idx), len(label_arr), h5))
                continue

            missing_frames = np.where(np.isnan(idx))[0]
            trim_idx = idx[~np.isnan(idx)].astype('int32')
            label_arr = label_arr[~np.isnan(idx)]
        else:
            missing_frames = None
            trim_idx = np.arange(len(label_arr))

        # do we need the trim_idx here actually?
        match_idx = trim_idx[np.where(label_arr == syllable)[0]]
        breakpoints = np.where(np.diff(match_idx, axis=0) > 1)[0]

        if len(breakpoints) < 1:
            continue

        breakpoints = zip(np.r_[0, breakpoints+1], np.r_[breakpoints, len(match_idx)-1])
        for i, j in breakpoints:
            # strike out movies that have missing frames
            if missing_frames is not None:
                if np.any(np.logical_and(missing_frames >= i, missing_frames <= j)):
                    continue
            syllable_slices.append([(match_idx[i], match_idx[j] + 1), label_uuid, h5])

    return syllable_slices


@np_cache
def find_label_transitions(label_arr: Union[dict, np.ndarray]) -> np.ndarray:
    '''
    Finds indices where a label transitions into another label. This
    function is cached to increase performance because it is called frequently.
    Parameters
    ----------
    label_arr (dict or np.ndarray): list or dict of predicted syllable labels.

    Returns
    -------
    inds (np.ndarray): Array of syllable transition indices for each session uuid.
    '''

    if isinstance(label_arr, dict):
        return valmap(find_label_transitions, label_arr)
    elif isinstance(label_arr, np.ndarray):
        inds = np.where(np.diff(label_arr) != 0)[0] + 1
        return inds
    else:
        raise TypeError('passed the wrong datatype')


def compress_label_sequence(label_arr: Union[dict, np.ndarray]) -> np.ndarray:
    '''
    Removes repeating values from a label sequence. It assumes the first
    label is '-5', which is unused for behavioral analysis, and removes it.
    Parameters
    ----------
    label_arr (dict or np.ndarray): list or dict of predicted syllable labels.

    Returns
    -------
    label_arr[inds] (dict or np.ndarray): the compressed version of the label arrays.
    '''

    if isinstance(label_arr, dict):
        return valmap(compress_label_sequence, label_arr)
    elif isinstance(label_arr, np.ndarray):
        inds = find_label_transitions(label_arr)
        return label_arr[inds]
    else:
        raise TypeError('passed the wrong datatype')


def calculate_label_durations(label_arr: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
    '''
    Calculates syllable label durations.
    Parameters
    ----------
    label_arr (dict or np.ndarray): list or dict of predicted syllable labels.

    Returns
    -------
    np.diffs(inds) (np.ndarray): list of durations for each syllable in respective label order.
    '''

    if isinstance(label_arr, dict):
        return valmap(calculate_label_durations, label_arr)
    elif isinstance(label_arr, np.ndarray):
        tmp = np.concatenate((label_arr, [-5]))
        inds = find_label_transitions(tmp)
        return np.diff(inds)


def calculate_syllable_usage(labels: Union[dict, pd.DataFrame]):
    '''
    Calculates a dictionary of uuid to syllable usage key-values pairs.
    Parameters
    ----------
    label_arr (dict or pd.DataFrame): list or DataFrame of predicted syllable labels.

    Returns
    -------
    (dict): dictionary of syllable usage probabilities.
    '''

    if isinstance(labels, pd.DataFrame):
        usage_df = labels.syllable.value_counts()
    elif isinstance(labels, (dict, OrderedDict)):
        syllables = concat(compress_label_sequence(labels).values())
        usage_df = pd.Series(syllables).value_counts()
    return dict(zip(usage_df.index.to_numpy(), usage_df.to_numpy()))


def get_syllable_statistics(data, fill_value=-5, max_syllable=100, count='usage'):
    '''
    Compute the syllable statistics from a set of model labels
    Parameters
    ----------
    data (list of np.array of ints): labels loaded from a model fit.
    fill_value (int): lagged label values in the labels array to remove.
    max_syllable (int): maximum syllable to consider.
    count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames).

    Returns
    -------
    usages (defaultdict): default dictionary of usages
    durations (defaultdict): default dictionary of durations
    '''

    usages = defaultdict(int)
    durations = defaultdict(list)

    if count == 'usage':
        use_usage = True
    elif count == 'frames':
        use_usage = False
    else:
        print('Inputted count is incorrect or not supported. Use "usage" or "frames".')
        print('Calculating statistics by syllable usage')
        use_usage = True

    for s in range(max_syllable):
        usages[s] = 0
        durations[s] = []

    if type(data) is list or (type(data) is np.ndarray and data.dtype == np.object):

        for v in data:
            seq_array, locs = _get_transitions(v)
            to_rem = np.where(np.logical_or(seq_array > max_syllable,
                                            seq_array == fill_value))

            seq_array = np.delete(seq_array, to_rem)
            locs = np.delete(locs, to_rem)
            durs = np.diff(np.insert(locs, len(locs), len(v)))

            for s, d in zip(seq_array, durs):
                if use_usage:
                    usages[s] = usages[s] + 1
                else:
                    usages[s] = usages[s] + d
                durations[s].append(d)

    else:#elif type(data) is np.ndarray and data.dtype == 'int16':

        seq_array, locs = _get_transitions(data)
        to_rem = np.where(seq_array > max_syllable)[0]

        seq_array = np.delete(seq_array, to_rem)
        locs = np.delete(locs, to_rem)
        durs = np.diff(np.insert(locs, len(locs), len(data)))

        for s, d in zip(seq_array, durs):
            if use_usage:
                usages[s] = usages[s] + 1
            else:
                usages[s] = usages[s] + d
            durations[s].append(d)


    usages = OrderedDict(sorted(usages.items()))
    durations = OrderedDict(sorted(durations.items()))


    return usages, durations


def labels_to_changepoints(labels, fs=30.):
    '''
    Compute the transition matrix from a set of model labels.
    Parameters
    ----------
    labels (list of np.array of ints): labels loaded from a model fit.
    fs (float): sampling rate of camera.

    Returns
    -------
    cp_dist (list of np.array of floats): list of block durations per element in labels list.
    '''

    cp_dist = []

    for lab in labels:
        cp_dist.append(np.diff(_get_transitions(lab)[1].squeeze()) / fs)

    return np.concatenate(cp_dist)


def parse_batch_modeling(filename):
    '''
    Reads model parameter scan training results into a single dictionary.
    Parameters
    ----------
    filename (str): path to h5 manifest file containing all the model results.

    Returns
    -------
    results_dict (dict): dictionary containing each model's training results,
    concatenated into a single list. Maintaining the original structure as though
    it was a single model's results.
    '''

    with h5py.File(filename, 'r') as f:
        scans = h5_to_dict(f, 'scans')
        params = h5_to_dict(f, 'metadata/parameters')
        results_dict = {
            'heldouts': np.squeeze(f['metadata/heldout_ll'][()]),
            'parameters': params,
            'scans': scans,
            'filenames': [join(dirname(filename), basename(fname.decode('utf-8')))
                          for fname in f['filenames']],
            'labels': np.squeeze(f['labels'][()]),
            'loglikes': np.squeeze(f['metadata/loglikes'][()]),
            'label_uuids': [str(_, 'utf-8') for _ in f['/metadata/train_list']]
        }

        results_dict['scan_parameters'] = dict((x, get(x, params, None)) for x in scans)

    return results_dict


def parse_model_results(model_obj, restart_idx=0, resample_idx=-1,
                        map_uuid_to_keys: bool = False,
                        sort_labels_by_usage: bool = False,
                        count: str = 'usage') -> dict:

    '''

    Parameters
    ----------
    model_obj (str or results returned from joblib.load): path to the model fit or a loaded model fit
    restart_idx (int): Select which model restart to load. (Only change for models with multiple restarts used)
    resample_idx (int): Indicates the parsing method according to the shape of the labels array.
    map_uuid_to_keys (bool): for labels, make a dictionary where each key, value pair
    contains the uuid and the labels for that session.
    sort_labels_by_usage (bool): sort labels by their usages.
    count (str): how to count syllable usage, either by number of emissions (usage),
    or number of frames (frames).

    Returns
    -------
    output_dict (dict): dictionary with labels and model parameters
    '''

    # reformat labels into something useful

    if type(model_obj) is str and (model_obj.endswith('.p') or model_obj.endswith('.pz')):
        model_obj = joblib.load(model_obj)
    elif type(model_obj) is str:
        raise RuntimeError('Can only parse models saved using joblib that end with .p or .pz')

    output_dict = deepcopy(model_obj)
    if type(output_dict['labels']) is list and type(output_dict['labels'][0]) is list:
        if np.ndim(output_dict['labels'][0][0]) == 2:
            output_dict['labels'] = [np.squeeze(tmp[resample_idx]) for tmp in output_dict['labels'][restart_idx]]
        elif np.ndim(output_dict['labels'][0][0]) == 1:
            output_dict['labels'] = [np.squeeze(tmp) for tmp in output_dict['labels'][restart_idx]]
        else:
            raise RuntimeError('Could not parse model labels')

    if type(output_dict['model_parameters']) is list:
        output_dict['model_parameters'] = output_dict['model_parameters'][restart_idx]

    if sort_labels_by_usage:
        output_dict['labels'], sorting = relabel_by_usage(output_dict['labels'], count=count)
        old_ar_mat = deepcopy(output_dict['model_parameters']['ar_mat'])
        old_nu = deepcopy(output_dict['model_parameters']['nu'])
        for i, sort_idx in enumerate(sorting):
            output_dict['model_parameters']['ar_mat'][i] = old_ar_mat[sort_idx]
            if type(output_dict['model_parameters']['nu']) is list:
                output_dict['model_parameters']['nu'][i] = old_nu[sort_idx]

    if map_uuid_to_keys:
        if 'train_list' in output_dict.keys():
            label_uuids = output_dict['train_list']
        else:
            label_uuids = output_dict['keys']

        label_dict = {uuid: lbl for uuid, lbl in zip(label_uuids, output_dict['labels'])}
        output_dict['labels'] = label_dict

    return output_dict


def relabel_by_usage(labels, fill_value=-5, count='usage'):
    '''
    Resort model labels by their usages.
    Parameters
    ----------
    labels (list of np.array of ints): labels loaded from a model fit
    fill_value (int): value prepended to modeling results to account for nlags
    count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames)

    Returns
    -------
    labels (list of np.array of ints): labels resorted by usage
    sorting (list): the new label sorting. The index corresponds to the new label,
    while the value corresponds to the old label.
    '''

    sorted_labels = deepcopy(labels)
    usages, _ = get_syllable_statistics(labels, fill_value=fill_value, count=count)
    sorting = []

    for w in sorted(usages, key=usages.get, reverse=True):
        sorting.append(w)

    for i, v in enumerate(labels):
        for j, idx in enumerate(sorting):
            sorted_labels[i][np.where(v == idx)] = j

    return sorted_labels, sorting


def results_to_dataframe(model_dict, index_dict, sort=False, count='usage', normalize=True, max_syllable=40,
                         include_meta=['SessionName', 'SubjectName', 'StartTime']):
    '''

    Parameters
    ----------
    model_dict (dict): loaded model results dictionary.
    index_dict (dict): loaded index file dictionary
    sort (bool): indicate whether to relabel syllables by usage.
    count (str): indicate what to sort the labels by: usage, or frames
    normalize (bool): unused.
    max_syllable (int): maximum number of syllables to include in dataframe.
    include_meta (list): mouse metadata to include in dataframe.

    Returns
    -------
    df (pd.DataFrame): DataFrame containing model results and metadata.
    df_dict (dict): dictionary representation of the DataFrame.
    '''

    if type(model_dict) is str:
        model_dict = parse_model_results(model_dict)

    if sort:
        model_dict['labels'] = relabel_by_usage(model_dict['labels'], count=count)[0]
    # by default the keys are the uuids

    if 'train_list' in model_dict.keys():
        label_uuids = model_dict['train_list']
    else:
        label_uuids = model_dict['keys']

    # durations = []

    df_dict = {
            'usage': [],
            'group': [],
            'syllable': []
        }

    for key in include_meta:
        df_dict[key] = []

    try:
        groups = [index_dict['files'][uuid]['group'] for uuid in label_uuids]
    except:
        groups = []
        for i, uuid in enumerate(label_uuids):
            groups.append(index_dict['files'][i].get('group'))

    try:
        metadata = [index_dict['files'][uuid]['metadata'] for uuid in label_uuids]
    except:
        metadata = []
        for i, uuid in enumerate(label_uuids):
            metadata.append(index_dict['files'][i].get('metadata'))

    for i, label_arr in enumerate(model_dict['labels']):
        tmp_usages, tmp_durations = get_syllable_statistics(label_arr, count=count, max_syllable=max_syllable)
        total_usage = np.sum(list(tmp_usages.values()))
        if total_usage <= 0:
            total_usage = 1.0
        for k, v in tmp_usages.items():
            df_dict['usage'].append(v / total_usage)
            df_dict['syllable'].append(k)
            df_dict['group'].append(groups[i])

            for meta_key in include_meta:
                df_dict[meta_key].append(metadata[i][meta_key])

    df = pd.DataFrame.from_dict(data=df_dict)

    return df, df_dict

def simulate_ar_trajectory(ar_mat, init_points=None, sim_points=100):
    '''
    Simulate auto-regressive trajectory matrices from
    optionally randomly projected initalized points.
    Parameters
    ----------
    ar_mat (np.ndarray): numpy array representing the autoregressive matrix of each model state.
    init_points (np.ndarray): pre-initialzed array of the same shape as the ar-matrices.
    sim_points (int): number of trajectories to simulate.

    Returns
    -------
    sim_mat[nlags:] simulated AR matrices excluding lagged values.
    '''

    npcs = ar_mat.shape[0]

    if ar_mat.shape[1] % npcs == 1:
        affine_term = ar_mat[:, -1]
        ar_mat = ar_mat[:, :-1]
    else:
        affine_term = np.zeros((ar_mat.shape[0], ), dtype='float32')

    nlags = ar_mat.shape[1] // npcs

    # print('Found {} pcs and {} lags in AR matrix'.format(npcs, nlags))

    if init_points is None:
        init_points = np.zeros((nlags, npcs), dtype='float32')

    sim_mat = np.zeros((sim_points + nlags, npcs), dtype='float32')
    sim_mat[:nlags] = init_points[:nlags]

    use_mat = np.zeros((nlags, npcs, npcs))

    for i in range(len(use_mat)):
        use_mat[i] = ar_mat[:, i * npcs: (i + 1) * npcs]

    for i in range(sim_points):
        sim_idx = i + nlags
        result = 0
        for j in range(1, nlags + 1):
            result += sim_mat[sim_idx - j].dot(use_mat[nlags - j])
        result += affine_term

        sim_mat[sim_idx, :] = result

    return sim_mat[nlags:]


def sort_batch_results(data, averaging=True, filenames=None, **kwargs):
    '''
    Sort modeling results from batch/parameter scan.
    Parameters
    ----------
    data (np.ndarray): model AR-matrices.
    averaging (bool): return an average of all the model AR-matrices.
    filenames (list): list of paths to fit models.
    kwargs (dict): dict of extra keyword arguments.

    Returns
    -------
    new_matrix (np.ndarray):
    param_dict (dict):
    filename_index (list): list of filenames associated with each model.
    '''

    parameters = np.hstack(kwargs.values())
    param_sets = np.unique(parameters, axis=0)
    param_dict = {k: np.unique(v[np.isfinite(v)]) for k, v in kwargs.items()}

    param_list = list(param_dict.values())
    param_list = [p[np.isfinite(p)] for p in param_list]
    new_shape = tuple([len(v) for v in param_list])

    if filenames is not None:
        filename_index = np.empty(new_shape, dtype=np.object)
        for i, v in np.ndenumerate(filename_index):
            filename_index[i] = []
    else:
        filename_index = None

    dims = len(new_shape)

    if dims > 2:
        raise NotImplementedError('No support for more than 2 dimensions')

    if averaging:
        new_matrix = np.zeros(new_shape, dtype=data.dtype)
        new_count = np.zeros(new_shape, dtype=data.dtype)
    else:
        _, cnts = np.unique(parameters, return_counts=True, axis=0)
        nrestarts = cnts.max()
        if nrestarts == 0:
            raise RuntimeError('Did not detect any restarts')

        new_shape = tuple([nrestarts]) + new_shape
        new_matrix = np.zeros(new_shape, dtype=data.dtype)
        new_matrix[:] = np.nan

    # TODO: add support for no averaging (just default_dict or list)

    for param in param_sets:
        row_matches = np.where((parameters == param).all(axis=1))[0]
        idx = np.zeros((len(param),), dtype='int')

        if np.any(np.isnan(param)):
            continue

        for i, p in enumerate(param):
            idx[i] = int(np.where(param_list[i] == p)[0])

        for i, row in enumerate(row_matches):
            if dims == 2:
                if idx[0] >= 0 and idx[1] >= 0:
                    if averaging:
                        new_matrix[idx[0], idx[1]] = np.nansum([new_matrix[idx[0], idx[1]], data[row]])
                        new_count[idx[0], idx[1]] += 1
                    else:
                        new_matrix[i, idx[0], idx[1]] = data[row]
                    if filenames is not None:
                        filename_index[idx[0], idx[1]].append(filenames[row])
            elif dims == 1:
                if idx >= 0:
                    if averaging:
                        new_matrix[idx] = np.nansum([new_matrix[idx], data[row]])
                        new_count[idx] += 1
                    else:
                        new_matrix[i, idx] = data[row]
                    if filenames is not None:
                        filename_index[idx].append(filenames[row])

    if averaging:
        new_matrix[new_count == 0] = np.nan
        new_matrix /= new_count

    return new_matrix, param_dict, filename_index


def whiten_pcs(pca_scores, method='all', center=True):
    """Whiten PC scores using Cholesky whitening

    Args:
        pca_scores (dict): dictionary where values are pca_scores (2d np arrays)
        method (str): 'all' to whiten using the covariance estimated from all keys, or 'each' to whiten each separately
        center (bool): whether or not to center the data

    Returns:
        whitened_scores (dict): dictionary of whitened pc scores

    Examples:

        Load in pca_scores and whiten

        >> from moseq2_viz.util import h5_to_dict
        >> from moseq2_viz.model.util import whiten_pcs
        >> pca_scores = h5_to_dict('pca_scores.h5', '/scores')
        >> whitened_scores = whiten_pcs(pca_scores, method='all')

    """

    if method[0].lower() == 'a':
        whitened_scores = _whiten_all(pca_scores)
    else:
        whitened_scores = {}
        for k, v in pca_scores.items():
            whitened_scores[k] = _whiten_all({k: v})[k]

    return whitened_scores


def normalize_pcs(pca_scores: dict, method: str = 'z') -> dict:
    '''
    Normalize PC scores. Options are: demean, zscore, ind-zscore.
    demean: subtract the mean from each score.
    Parameters
    ----------
    pca_scores (dict): dict of uuid to PC-scores key-value pairs.
    method (str): the type of normalization to perform (demean, zscore, ind-zscore)

    Returns
    -------
    norm_scores (dict): a dictionary of normalized PC scores.
    '''

    norm_scores = deepcopy(pca_scores)
    if method.lower()[0] == 'z':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig
    elif method.lower()[0] == 'm':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = v - mu
    elif method == 'ind-zscore':
        for k, v in norm_scores.items():
            norm_scores[k] = (v - np.nanmean(v)) / np.nanstd(v)
    else:
        print('Using default: z-score')
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig


    return norm_scores


def _gen_to_arr(generator: Iterator[Any]) -> np.ndarray:
    '''
    Cast a generator object into a numpy array.
    Parameters
    ----------
    generator (Iterator[Any]): a generator object.

    Returns
    -------
    np.array(list(generator)) (np.array): numpy array of generated list.
    '''

    return np.array(list(generator))


def retrieve_pcs_from_slices(slices, pca_scores, max_dur=60, min_dur=3,
                             max_samples=100, npcs=10, subsampling=None,
                             remove_offset=False, **kwargs):
    '''

    Parameters
    ----------
    slices (np.ndarray):
    pca_scores (np.ndarray):
    max_dur (int): maximum slice length.
    min_dur (int): minimum slice length.
    max_samples (int): maximum number of samples to slices to retrieve.
    npcs (int): number of pcs to use.
    subsampling (int): number of neighboring PCs to subsample from.
    remove_offset (bool): indicate whether to remove lag values.
    kwargs (dict): unused.

    Returns
    -------
    syllable_matrix (np.ndarray): 3D matrix of subsampled PC projected syllable slices.
    '''

    # pad using zeros, get dtw distances...

    # make function to filter syll durations
    filter_dur = compose(lambda dur: (dur < max_dur) & (dur > min_dur),
                         lambda inds: inds[1] - inds[0],
                         get(0))
    filtered_slices = _gen_to_arr(filter(filter_dur, slices))
    # select random samples
    inds = np.random.randint(0, len(filtered_slices), size=max_samples)
    use_slices = filtered_slices[inds]

    syllable_matrix = np.zeros((len(use_slices), max_dur, npcs), 'float32')

    for i, (idx, uuid, _) in enumerate(use_slices):
        syllable_matrix[i, :idx[1]-idx[0], :] = pca_scores[uuid][idx[0]:idx[1], :npcs]

    if remove_offset:
        syllable_matrix = syllable_matrix - syllable_matrix[:, 0, :][:, None, :]

    # get cluster averages - really good at selecting for different durations of a syllable
    if subsampling is not None and subsampling > 0:
        try:
            km = KMeans(subsampling)
            syllable_matrix = syllable_matrix.reshape(syllable_matrix.shape[0], max_dur * npcs)
            syllable_matrix = syllable_matrix[np.all(~np.isnan(syllable_matrix), axis=1), :]
            km.fit(syllable_matrix)
            syllable_matrix = km.cluster_centers_.reshape(subsampling, max_dur, npcs)
        except Exception:
            syllable_matrix = np.zeros((subsampling, max_dur, npcs))
            syllable_matrix[:] = np.nan

    return syllable_matrix


'''
def relabel_by_usage(labels: Union[list, np.ndarray], fill_value: int = -5,
                     count: str = 'usage') -> Union[list, np.ndarray]:
    Re-sort model labels by their usages

    Args:
        labels: labels loaded from a model fit
        fill_value: value prepended to modeling results to account for nlags
        count: how to count syllable usage - either by emission number (usage) or number of frames (frames)

    Returns:
        labels: labels resorted by usage

    Examples:
        Load in model results and sort labels by usages::

            from moseq2_viz.model.util import parse_model_results, relabel_by_usage
            model_results = parse_model_results('mymodel.p')
            sorted_labels = relabel_by_usage(model_results['labels'], count='usage')


    if isinstance(labels, (list, np.ndarray)):
        return _relabel_list_by_usage(labels, fill_value=fill_value, count=count)
    elif isinstance(labels, dict):
        # rest assured, in python 3 dicts are ordered by default
        uuids = list(labels.keys())
        sorted_labels, sorting = _relabel_list_by_usage(list(labels.values()), fill_value=fill_value, count=count)
        return dict(zip(uuids, sorted_labels)), sorting
    else:
        raise ValueError(f'processing of datatype {type(labels)} not implemented or recognized')
'''