'''

Utility functions specifically responsible for handling model data during pre and post processing.

'''

import os
import h5py
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import starmap
from numpy import linalg as LA
from cytoolz.curried import get
from sklearn.cluster import KMeans
from typing import Iterator, Any, Dict, Union
from collections import defaultdict, OrderedDict
from scipy.optimize import linear_sum_assignment
from os.path import join, basename, dirname, exists
from moseq2_viz.util import h5_to_dict, star
from moseq2_viz.model.trans_graph import get_transitions
from cytoolz import curry, valmap, compose, complement, itemmap, concat, keyfilter


def merge_models(model_dir, ext='p',count='usage'):
    '''
    Merges model states by using the Hungarian Algorithm:
    a minimum distance state matching algorithm. User inputs a
    directory containing models to merge, (and the name of the latest-trained
    model) to match other model states to.

    Parameters
    ----------
    model_dir (str): path to directory containing all the models to merge.
    ext (str): model extension to search for.
    count (str): method to compute usages 'usage' or 'frames'.

    Returns
    -------
    model_data (dict): a dictionary containing all the new
    keys and state-matched labels.
    '''

    tmp = join(model_dir, '*.'+ext.strip('.'))
    model_paths = [m for m in glob.glob(tmp)]
    model_data = {}

    for m, model_fit in enumerate(model_paths):
        unit_data = parse_model_results(joblib.load(model_fit), \
                                        sort_labels_by_usage=True, count=count)
        for k, v in unit_data.items():
            if k not in list(model_data.keys()):
                model_data[k] = v
            else:
                if k == 'model_parameters':
                    try:
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
                        for session in unit_data['labels']:
                            for oldlbl in session:
                                try:
                                    adjusted_labels.append(mapping[oldlbl])
                                except:
                                    pass
                            model_data['labels'].append(np.array(adjusted_labels))
                    except:
                        print('Error, trying to merge models with unequal number of PCs. Skipping.')
                        pass
                elif k == 'keys' or k == 'train_list':
                    for i in v:
                        if i not in model_data[k]:
                            model_data[k].append(i)
                elif k == 'metadata':
                    for k1, v1 in unit_data[k].items():
                        if k1 == 'groups':
                            if isinstance(model_data[k][k1], dict):
                                model_data[k][k1].update(v1)
                            elif isinstance(model_data[k][k1], list):
                                model_data[k][k1] += v1
                        elif k1 == 'uuids':
                            for val in v1:
                                if val not in model_data[k][k1]:
                                    model_data[k][k1] = list(model_data[k][k1])
                                    model_data[k][k1].append(val)

    return model_data

def get_best_fit(cp_path, model_results):
    '''
    Returns the model with the closest median syllable duration to the PCA changepoints.

    Parameters
    ----------
    cp_path (str): Path to PCA Changepoints h5 file.
    model_results (dict): dict of pairs of model names paired with dict containing their respective changepoints.

    Returns
    -------
    best_model (str): Computed best-fit model key.
    pca_cps (1D array): 1-dimensional list of pc score block durations.
    '''

    # Load changepoints
    with h5py.File(cp_path, 'r') as f:
        cps = h5_to_dict(f, 'cps')

    # Compute changepoint distribution
    pca_cps = np.concatenate([np.diff(cp, axis=0) for k, cp in cps.items()]).flatten()
    
    best_model = None
    min_dist = np.inf
    
    for model in model_results.keys():
        # Load current models changepoints
        model_cps = model_results[model]['changepoints']
        try:
            # Get syllable duration difference
            syll_dur = abs(np.median(pca_cps) - np.median(model_cps))

            # If syllable duration is less than last saved value, update best model key
            if syll_dur < min_dist:
                min_dist = syll_dur
                best_model = model
        except:
            print('model doesnt match pcs')
            pass
        
    return best_model, pca_cps


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

def get_normalized_syllable_usages(model_data, max_syllable=100, count='usage'):
    '''
    Computes the overall syllable usages, and returns a 1D array of their corresponding usage values.

    Parameters
    ----------
    model_data (dict): dict object of modeling results
    max_syllable (int): number of syllables to compute the mean usage for.
    count (str): option for whether to count syllable usages; by 'frames', or 'usage'.

    Returns
    -------
    syllable_usages (1D np array): array of sorted syllable usages for all syllables in model
    '''

    # process the syllable usages over all frames/emissions in the entire cohort
    usages_by_mouse = np.array([list(get_syllable_statistics(data=labels,
                                                             fill_value=-5,
                                                             count=count,
                                                             max_syllable=max_syllable)[0].values()) \
                                for labels in model_data['labels']])

    syllable_usages = np.sum(usages_by_mouse, axis=0) / np.sum(usages_by_mouse)

    return syllable_usages

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
    any_nan = compose(np.any, np.isnan)
    non_nan = complement(any_nan)

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
    return vals


@curry
def get_syllable_slices(syllable, labels, label_uuids, index, trim_nans: bool = True) -> list:
    '''
    Get the indices that correspond to a specific syllable for each animal in a modeling run.

    Parameters
    ----------
    syllable (int): syllable number to get slices of.
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

    if isinstance(index['files'], (dict, OrderedDict)):
        h5s = {k: v['path'][0] for k, v in index['files'].items()}
    elif isinstance(index['files'], (tuple, list, np.ndarray)):
        h5s = {v['uuid']: v['path'][0] for v in index['files']}
    else:
        raise TypeError('"files" key in index not readable')

    # grab the original indices from the pca file as well...
    if trim_nans:
        with h5py.File(index['pca_path'], 'r') as f:
            score_idx = h5_to_dict(f, 'scores_idx')

    syllable_slices = []

    for label_arr, label_uuid in zip(labels, label_uuids):
        h5 = h5s[label_uuid]

        if trim_nans:
            idx = score_idx[label_uuid]

            if len(idx) > len(label_arr):
                warnings.warn(f'Index length {len(idx)} and label array length {len(label_arr)} in {h5}.'
                               ' Setting index length to label array length.')
                idx = idx[:len(label_arr)]
            elif len(idx) < len(label_arr):
                warnings.warn(f'Index length {len(idx)} and label array length {len(label_arr)} in {h5}.'
                               ' Skipping trim for this session.')
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


def find_label_transitions(label_arr: Union[dict, np.ndarray, pd.Series]) -> np.ndarray:
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
    elif isinstance(label_arr, (np.ndarray, pd.Series)):
        inds = np.where(np.diff(label_arr) != 0)[0] + 1
        return inds
    else:
        raise TypeError('passed the wrong datatype')


def compress_label_sequence(label_arr: Union[dict, np.ndarray, pd.Series]) -> np.ndarray:
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
    elif isinstance(label_arr, (np.ndarray, pd.Series)):
        inds = find_label_transitions(label_arr)
        return label_arr[inds]
    else:
        raise TypeError('passed the wrong datatype')


def calculate_syllable_durations(label_arr: Union[dict, np.ndarray]) -> Union[dict, np.ndarray]:
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
        return valmap(calculate_syllable_durations, label_arr)
    elif isinstance(label_arr, np.ndarray):
        tmp = np.concatenate((label_arr, [-5]))
        inds = find_label_transitions(tmp)
        return np.diff(inds)


def syll_onset(labels: np.ndarray) -> np.ndarray:
    '''
    Finds indices of syllable onsets.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

    Returns
    -------
    indices (np.ndarray): an array of indices denoting the beginning of each syllables.
    '''

    change = np.diff(labels) != 0
    indices = np.where(change)[0]
    indices += 1
    indices = np.concatenate(([0], indices))
    return indices


def syll_duration(labels: np.ndarray) -> np.ndarray:
    '''
    Computes the duration of each syllable.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

    Returns
    -------
    durations (np.ndarray): array of syllable durations.
    '''

    onsets = np.concatenate((syll_onset(labels), [labels.size]))
    durations = np.diff(onsets)
    return durations


def syll_id(labels: np.ndarray) -> np.ndarray:
    '''
    Returns the syllable label at each syllable transition.

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.

    Returns
    -------
    labels[onsets] (np.ndarray): an array of compressed labels.
    '''

    onsets = syll_onset(labels)
    return labels[onsets]


def get_syllable_usages(data, max_syllable=100, count='usage'):

    def _convert_to_usage(arr):
        if count == 'usage':
            arr = syll_id(arr)
        return pd.Series(arr).value_counts().reindex(range(max_syllable)).fillna(0)

    # a list of sessions
    if isinstance(data, list) and isinstance(data[0], (list, np.ndarray)):
        usages = sum(map(_convert_to_usage, data))
    elif isinstance(data, dict):
        usages = sum(map(_convert_to_usage, data.values()))
    else:
        raise TypeError('could not understand data parameter. Needs to be list or dict of labels')
    
    return dict(usages)


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
    usages (OrderedDict): default dictionary of usages
    durations (OrderedDict): default dictionary of durations
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

    if isinstance(data, list) or (isinstance(data, np.ndarray) and data.dtype == np.object):

        for v in data:
            seq_array, locs = get_transitions(v)
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

        seq_array, locs = get_transitions(data)
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


    usages = OrderedDict(sorted(usages.items())[:max_syllable])
    durations = OrderedDict(sorted(durations.items())[:max_syllable])


    return usages, durations


def labels_to_changepoints(labels, fs=30):
    '''
    Compute syllable durations and combine into a "changepoint" distribution.

    Parameters
    ----------
    labels (list of np.ndarray of ints): labels loaded from a model fit.
    fs (float): sampling rate of camera.

    Returns
    -------
    cp_dist (np.ndarray of floats): list of block durations per element in labels list.
    '''

    cp_dist = []

    for lab in labels:
        cp_dist.append(np.diff(get_transitions(lab)[1].squeeze()) / fs)

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
            'label_uuids': [str(_, 'utf-8') for _ in f['/metadata/train_list']],
            'scan_parameters': dict((x, get(x, params, None)) for x in scans)
        }

    return results_dict


def parse_model_results(model_obj, restart_idx=0, resample_idx=-1,
                        map_uuid_to_keys: bool = False,
                        sort_labels_by_usage: bool = False,
                        count: str = 'usage') -> dict:

    '''
    Reads model file and returns dictionary containing modeled results and some metadata.

    Parameters
    ----------
    model_obj (str or results returned from joblib.load): path to the model fit or a loaded model fit
    restart_idx (int): Select which model restart to load. (Only change for models with multiple restarts used)
    resample_idx (int): Legacy parameter used to select labels from a specific sampling iteration. Default is the last iteration (-1)
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

    if isinstance(model_obj, dict):
        output_dict = deepcopy(model_obj)
    elif isinstance(model_obj, str) and model_obj.endswith(('.p', '.pz')):
        output_dict = joblib.load(model_obj)
    else:
        raise RuntimeError('Can only parse model paths saved using joblib that end with .p or .pz')

    # a bunch of legacy loading
    if isinstance(output_dict['labels'], list) and isinstance(output_dict['labels'][0], list):
        if np.ndim(output_dict['labels'][0][0]) == 2:
            output_dict['labels'] = [np.squeeze(tmp[resample_idx]) for tmp in output_dict['labels'][restart_idx]]
        elif np.ndim(output_dict['labels'][0][0]) == 1:
            output_dict['labels'] = [np.squeeze(tmp) for tmp in output_dict['labels'][restart_idx]]
        else:
            raise RuntimeError('Could not parse model labels')

    if isinstance(output_dict['model_parameters'], list):
        output_dict['model_parameters'] = output_dict['model_parameters'][restart_idx]

    if sort_labels_by_usage:
        output_dict['labels'], sorting = relabel_by_usage(output_dict['labels'], count=count)
        old_ar_mat = deepcopy(output_dict['model_parameters']['ar_mat'])
        old_nu = deepcopy(output_dict['model_parameters']['nu'])
        for i, sort_idx in enumerate(sorting):
            output_dict['model_parameters']['ar_mat'][i] = old_ar_mat[sort_idx]
            if isinstance(output_dict['model_parameters']['nu'], list):
                output_dict['model_parameters']['nu'][i] = old_nu[sort_idx]

    if map_uuid_to_keys:
        if 'train_list' in output_dict:
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
    labels (list or dict): label sequences loaded from a model fit
    fill_value (int): value prepended to modeling results to account for nlags
    count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames)

    Returns
    -------
    labels (list or dict): label sequences sorted by usage
    sorting (list): the new label sorting. The index corresponds to the new label,
    while the value corresponds to the old label.
    '''
    assert count in ('usage', 'frames'), 'count must be "usage" or "frames"'

    sorted_labels = deepcopy(labels)
    usages = get_syllable_usages(labels, count=count)

    sorting = sorted(usages, key=usages.get, reverse=True)

    if isinstance(labels, list):
        _iter = enumerate(labels)
    elif isinstance(labels, dict):
        _iter = labels.items()

    for i, v in _iter:
        for j, idx in enumerate(sorting):
            sorted_labels[i][np.where(v == idx)] = j

    return sorted_labels, sorting

# TODO: figure out the difference between this and `results_to_dataframe`
def get_frame_label_df(labels, uuids, groups):
    '''
    Returns a DataFrame with rows for each session, frame indices as columns,
    and syllable label values corresponding to these frames+sessions for each frame.

    Parameters
    ----------
    labels (2D np.array): list of np arrays containing syllable labels with respect to individually labeled frames.
     Index by uuids.
    uuids (list): list of uuid strings corresponding to each "row" in labels.
    groups (list): list of group strings corresponding to each "row" in labels.

    Returns
    -------
    label_df (pd.DataFrame): Dataframe of shape (nsessions x max(len(labels))). At columns exceeding session's labeled frame
    count, values will be np.NaN. Rows are indexed by Multi-Index([['group', 'uuid'],...)

    '''

    total_columns = max(map(len, labels))
    label_df = pd.DataFrame(labels, columns=range(total_columns), index=[groups, uuids])
    return label_df

def results_to_dataframe(model_dict, index_dict, sort=False, count='usage', max_syllable=40,
                         include_meta=['SessionName', 'SubjectName', 'StartTime'], compute_labels=False):
    '''
    Converts inputted model dictionary to DataFrame with user specified metadata columns.
    Also generates a DataFrame containing frame-by-frame syllable labels for all sessions.

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
    label_df (pd.DataFrame): DataFrame containing syllable labels at each frame (nsessions rows x max(nframes) cols)
    '''

    if type(model_dict) is str:
        model_dict = parse_model_results(model_dict)

    if sort:
        model_dict['labels'] = relabel_by_usage(model_dict['labels'], count=count)[0]

    # by default the keys are the uuids

    label_uuids = model_dict.get('train_list', model_dict['keys'])

    df_dict = {
        'usage': [],
        'duration': [],
        'uuid': [],
        'group': [],
        'syllable': []
    }

    for key in include_meta:
        df_dict[key] = []

    try:
        groups = [index_dict['files'][uuid].get('group', 'default') for uuid in label_uuids]
    except:
        groups = []
        try:
            for i, uuid in enumerate(label_uuids):
                groups.append(index_dict['files'][i].get('group', 'default'))
        except:
            print('WARNING: model results uuids do not match index file uuids.')
            groups.append('default')

    try:
        metadata = [index_dict['files'][uuid]['metadata'] for uuid in label_uuids]
    except:
        metadata = []
        for i, uuid in enumerate(label_uuids):
            metadata.append(index_dict['files'][i].get('metadata', {}))

    # get frame-by-frame label DataFrame
    if compute_labels:
        try:
            label_df = get_frame_label_df(model_dict['labels'], label_uuids, groups)
        except:
            label_df = []
            print('Could not compute frame-label dataframe.')
    else:
        label_df = []

    for i, label_arr in enumerate(model_dict['labels']):
        tmp_usages, tmp_durations = get_syllable_statistics(label_arr, count=count, max_syllable=max_syllable)
        total_usage = np.sum(list(tmp_usages.values()))
        if total_usage <= 0:
            total_usage = 1.0
        for k, v in tmp_usages.items():
            # average syll duration will be sum of all syllable durations divided by number of total sequences
            durs = tmp_durations[k]
            num_seqs = len(durs)
            if len(durs) == 0:
                num_seqs = 1.0

            df_dict['duration'].append(sum(durs) / num_seqs)
            df_dict['usage'].append(v / total_usage)
            df_dict['syllable'].append(k)
            df_dict['group'].append(groups[i])
            df_dict['uuid'].append(label_uuids[i])

            for meta_key in include_meta:
                df_dict[meta_key].append(metadata[i][meta_key])

    df = pd.DataFrame.from_dict(data=df_dict)

    return df, label_df


def compute_syllable_onset(labels):
    onset = pd.Series(labels).diff().fillna(1) != 0
    return onset.to_numpy()


def prepare_model_dataframe(model_path, pca_path):
    '''
    Creates a dataframe from syllable labels to be aligned with scalars.
    '''
    mdl = parse_model_results(model_path, map_uuid_to_keys=True)
    labels = mdl['labels']

    usage, _ = relabel_by_usage(labels, count='usage')
    frames, _ = relabel_by_usage(labels, count='frames')

    scores_idx = h5_to_dict(pca_path, path='scores_idx')

    # make sure all pcs align with labels
    if not all(len(labels[k]) == len(v) for k, v in scores_idx.items()):
        raise ValueError('PC scores don\'t align with labels')

    _df = pd.concat((pd.DataFrame({
        'uuid': k,
        'labels (original)': v,
        'labels (usage sort)': usage[k],
        'labels (frames sort)': frames[k],
        'onset': compute_syllable_onset(v),
        'frame index': scores_idx[k]
    }) for k, v in labels.items()), ignore_index=True)

    return _df


def simulate_ar_trajectory(ar_mat, init_points=None, sim_points=100):
    '''
    Simulate auto-regressive trajectory matrices from
    optionally randomly projected initalized points.

    Parameters
    ----------
    ar_mat (3D np.ndarray): numpy array representing the autoregressive matrix of each model state.
    init_points (2D np.ndarray): pre-initialzed array of the same shape as the ar-matrices.
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
    new_matrix (np.ndarray): either average of all AR-matrices, or top sorted matrix
    param_dict (dict): model parameter dict
    filename_index (list): list of filenames associated with each model.
    '''

    parameters = np.hstack(list(kwargs.values()))
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
    """
    Whiten PC scores using Cholesky whitening

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
        whitened_scores = _whiten_all(pca_scores, center=center)
    else:
        whitened_scores = {}
        for k, v in pca_scores.items():
            whitened_scores[k] = _whiten_all({k: v}, center=center)[k]

    return whitened_scores


def normalize_pcs(pca_scores: dict, method: str = 'zscore') -> dict:
    '''
    Normalize PC scores. Options are: demean, zscore, ind-zscore.
    zscore: standardize pc scores using all data
    demean: subtract the mean from each score.
    ind-zscore: zscore each session independently

    Parameters
    ----------
    pca_scores (dict): dict of uuid to PC-scores key-value pairs.
    method (str): the type of normalization to perform (demean, zscore, ind-zscore)

    Returns
    -------
    norm_scores (dict): a dictionary of normalized PC scores.
    '''
    if method not in ('zscore', 'demean', 'ind-zscore'):
        raise ValueError(f'normalization {method} not supported. Please use: "zscore", "demean", or "ind-zscore"')

    norm_scores = deepcopy(pca_scores)
    if method.lower() == 'zscore':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        norm_scores = valmap(lambda v: (v - mu) / sig, norm_scores)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig
    elif method.lower() == 'demean':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        norm_scores = valmap(lambda v: v - mu, norm_scores)
    elif method == 'ind-zscore':
        norm_scores = valmap(lambda v: (v - np.nanmean(v, axis=0)) / np.nanstd(v, axis=0), norm_scores)

    return norm_scores


def _gen_to_arr(generator: Iterator[Any]) -> np.ndarray:
    '''
    Cast a generator object into a numpy array.

    Parameters
    ----------
    generator (Iterator[Any]): a generator object.

    Returns
    -------
    arr (np.ndarray): numpy array of generated list.
    '''

    return np.array(list(generator))


def retrieve_pcs_from_slices(slices, pca_scores, max_dur=60, min_dur=3,
                             max_samples=100, npcs=10, subsampling=None,
                             remove_offset=False, **kwargs):
    '''
    Subsample Principal components from syllable slices

    Parameters
    ----------
    slices (np.ndarray): syllable slice or subarray to compute PCs for
    pca_scores (np.ndarray): PC scores for respective session.
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


def make_separate_crowd_movies(config_data, sorted_index, group_keys, label_dict, output_dir, ordering, sessions=False):
    '''
    Helper function that writes syllable crowd movies for each given grouping found in group_keys, and returns
     a dictionary with session/group name keys paired with paths to their respective generated crowd movies.

    Parameters
    ----------
    config_data (dict): Loaded crowd movie writing configuration parameters.
    sorted_index (dict): Loaded index file and sorted files in list.
    group_keys (dict): Dict of group/session name keys paired with UUIDS to match with labels.
    labels (2d list): list of syllable label lists for all sessions.
    label_uuids (list): list of corresponding session UUIDs for all sessions included in labels.
    output_dir (str): Path to output directory to save crowd movies in.
    ordering (list): ordering for the new mapping of the relabeled syllable usages.
    sessions (bool): indicates whether session crowd movies are being generated.

    Returns
    -------
    cm_paths (dict): group/session name keys paired with paths to their respectively generated syllable crowd movies.
    '''

    from moseq2_viz.io.video import write_crowd_movies

    cm_paths = {}
    for k, uuids in group_keys.items():
        # Filter group labels to pair with respective UUIDs
        labels = [label_dict[uuid] for uuid in uuids]

        # Get subset of sorted_index including only included session sources
        group_index = {
            'files': keyfilter(lambda k: k in uuids, sorted_index['files']),
            'pca_path': sorted_index['pca_path']
        }

        # create a subdirectory for each group
        output_subdir = join(output_dir, k)
        os.makedirs(output_subdir, exist_ok=True)

        # Write crowd movie for given group and syllable(s)
        cm_paths[k] = write_crowd_movies(group_index, config_data, ordering,
                                         labels, uuids, output_subdir)

    return cm_paths