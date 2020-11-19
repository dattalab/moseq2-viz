'''

Syllable label information utility functions.
Contains duplicate functions from moseq2-model + additional syllable sorting/relabeling functions.

'''

import numpy as np
import pandas as pd
from moseq2_viz.model.util import syll_duration, syll_id, syll_onset


def labels_to_df(labels, uuid) -> pd.DataFrame:
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


def sort_syllables_by_stat_difference(complete_df, ctrl_group, exp_group, max_sylls=None, stat='usage'):
    '''
    Computes the syllable ordering for the difference of the inputted groups (exp - ctrl).
    The sorted result will yield an array will indices depicting the largest positive (upregulated)
    difference between exp and ctrl groups on the left, and vice versa on the right.

    Parameters
    ----------
    complete_df (pd.DataFrame): dataframe containing the statistical information about syllable data [usages, durs, etc.]
    ctrl_group (str): Control group.
    exp_group (str): Experimental group.
    max_sylls (int): maximum number of syllables to include in ordering.
    stat (str): choice of statistic to order mutations by: {usage, duration, speed}.

    Returns
    -------
    mutation_ordering (list): list of array indices for the new label mapping.
    '''

    # Prepare DataFrame
    mutation_df = complete_df.groupby(['group', 'syllable'], as_index=False).mean()

    # Get groups to measure mutation by
    control_df = mutation_df[mutation_df['group'] == ctrl_group]
    exp_df = mutation_df[mutation_df['group'] == exp_group]

    # compute mean difference at each syll usage
    diff_df = exp_df.groupby('syllable', as_index=True).mean() \
        .sub(control_df.groupby('syllable', as_index=True).mean(), fill_value=0)

    # sort them from most mutant to least mutant
    mutation_ordering = diff_df.sort_values(by=stat, ascending=False).index

    if max_sylls is not None:
        mutation_ordering = mutation_ordering[:max_sylls]

    return mutation_ordering


def sort_syllables_by_stat(complete_df, stat='usage', max_sylls=None):
    '''
    Computes the sorted ordering of the given DataFrame with respect to the chosen stat.

    Parameters
    ----------
    complete_df (pd.DataFrame): DataFrame containing the statistical information about syllable data [usages, durs, etc.]
    stat (str): choice of statistic to order syllables by: {usage, duration, speed}.
    max_sylls (int or None): maximum number of syllables to include in ordering

    Returns
    -------
    ordering (list): list of sorted syllables by stat.
    relabel_mapping (dict): a dict with key-value pairs {old_ordering: new_ordering}.
    '''

    tmp = complete_df.groupby('syllable', as_index=False).mean()
    tmp = tmp.sort_values(by=stat, ascending=False)

    # Get sorted ordering
    ordering = tmp.index.to_numpy()

    if max_sylls is not None:
        ordering = ordering[:max_sylls]

    # Get order mapping
    relabel_mapping = {o: i for i, o in enumerate(ordering)}

    return ordering, relabel_mapping