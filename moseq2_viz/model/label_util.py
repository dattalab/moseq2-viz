'''

Syllable label information utility functions.

Contains duplicate functions from moseq2-model + additional syllable sorting functions.

'''

import pandas as pd
import numpy as np

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


def to_df(labels, uuid) -> pd.DataFrame:
    '''
    Convert labels numpy.ndarray to pandas.DataFrame

    Parameters
    ----------
    labels (np.ndarray): array of syllable labels for a mouse.
    uuid (list): list of session uuids representing each series of labels.

    Returns
    -------
    df (pd.DataFrame): DataFrame of syllables, durations, onsets, and session uuids.
    '''

    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    elif isinstance(labels, list):
        labels = np.array(labels, dtype=np.int32)

    df = pd.DataFrame({
        'syll': syll_id(labels),
        'dur': syll_duration(labels),
        'onset': syll_onset(labels),
        'uuid': uuid
    })

    return df


def get_syllable_muteness_ordering(complete_df, ctrl_group, exp_group, max_sylls=None, stat='usage'):
    '''
    Computes the syllable ordering for the difference of the inputted groups (exp - ctrl).
    The sorted result will yield an array will indices depicting the largest positive (upregulated)
    difference between exp and ctrl groups on the left, and vice versa on the right.

    Parameters
    ----------
    complete_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
    ctrl_group (str): Control group.
    exp_group (str): Experimental group.
    max_sylls (str): maximum number of syllables to include in ordering.
    stat (str): choice of statistic to order mutations by: {usage, duration, speed}.

    Returns
    -------
    muteness_ordering (list): list of array indices for the new label mapping.
    '''

    muteness_df = complete_df.groupby(['group', 'syllable'], as_index=False).mean()

    control_df = muteness_df[muteness_df['group'] == ctrl_group]
    exp_df = muteness_df[muteness_df['group'] == exp_group]

    # compute mean difference at each syll usage
    diff_df = exp_df.groupby('syllable', as_index=True).mean() \
        .sub(control_df.groupby('syllable', as_index=True).mean(), fill_value=0)
    if max_sylls == None:
        max_sylls = len(diff_df)

    # sort them from most mutant to least mutant
    muteness_ordering = diff_df.sort_values(by=stat, ascending=False).index[:max_sylls+1] # adding 1 for sylls [0-40] (for example)

    return muteness_ordering


def get_sorted_syllable_stat_ordering(complete_df, stat='usage'):
    '''
    Computes the sorted ordering of the given DataFrame with respect to the chosen stat.

    Parameters
    ----------
    complete_df (pd.DataFrame): DataFrame containing the statistical information about syllable data [usages, durs, etc.]
    stat (str): choice of statistic to order mutations by: {usage, duration, speed}.

    Returns
    -------
    ordering (list): list of newly mapped array (syllable label) indices.
    relabel_mapping (dict): dict of mappings from old to (descending-order y-label sorting) and ascending-order x range.
    '''

    tmp = complete_df.groupby(['syllable'], as_index=False).mean().copy()
    tmp.sort_values(by=[stat], inplace=True, ascending=False)

    ordering = tmp.syllable.to_numpy()
    relabel_mapping = {o: i for i, o in enumerate(ordering)}

    return ordering, relabel_mapping