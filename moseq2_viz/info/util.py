'''
Utility functions for computing syllable usage entropy, and syllable transition entropy rate.
These can be used for measuring modeling model performance and group separability.
'''

import numpy as np
from moseq2_viz.model.util import get_syllable_statistics, get_transition_matrix, relabel_by_usage


def entropy(labels, truncate_syllable=40, smoothing=1.0,
            relabel_by='usage'):
    '''
    Computes syllable usage entropy, base 2.

    Parameters
    ----------
    labels (np.ndarray): array of predicted syllable labels
    truncate_syllable (int): truncate list of relabeled syllables
    smoothing (float): a constant added to label usages before normalization
    relabel_by (str): mode to relabel predicted labels.

    Returns
    -------
    ent (list): list of entropy values for each syllable label.
    '''

    # Compute relabeled usages
    labels, _ = relabel_by_usage(labels, count=relabel_by)

    ent = []
    for v in labels:
        # Get usage for given syllable
        usages = get_syllable_statistics([v])[0]

        # Truncate list to max number of included syllables
        syllables = np.array(list(usages.keys()))
        truncate_point = np.where(syllables == truncate_syllable)[0]

        # Get syllable truncate index if it wasn't found
        if truncate_point is None or len(truncate_point) != 1:
            truncate_point = len(syllables)
        else:
            truncate_point = truncate_point[0]

        # truncate syllable list
        syllables = syllables[:truncate_point]

        # Compute average usage
        usages = np.array(list(usages.values()), dtype='float')
        usages = usages[:truncate_point] + smoothing
        usages /= usages.sum()

        # Compute entropy and append to syllable entropy list
        ent.append(-np.sum(usages * np.log2(usages)))

    return ent


def entropy_rate(labels, truncate_syllable=40, normalize='bigram',
                 smoothing=1.0, tm_smoothing=1.0, relabel_by='usage'):
    '''
    Computes entropy rate, base 2 using provided syllable labels. If
    syllable labels have not been re-labeled by usage, this function will do so.

    Parameters
    ----------
    labels (list or np.ndarray): a list of label arrays, where each entry in the list
            is an array of labels for one subject.
    truncate_syllable (int): the number of labels to keep for this calculation
    normalize (str): the type of transition matrix normalization to perform. Options
            are: 'bigram', 'rows', or 'columns'.
    smoothing (float): a constant added to label usages before normalization
    tm_smoothing (float): a constant added to label transtition counts before
            normalization.
    relabel_by (str): how to re-order labels. Options are: 'usage' and 'frames'.

    Returns
    -------
    ent (list): list of entropy rates per syllable label
    '''

    # Compute relabeled usages
    labels, _ = relabel_by_usage(labels, count=relabel_by)

    ent = []
    for v in labels:
        # Get usage for given syllable
        usages = get_syllable_statistics([v])[0]
        syllables = np.array(list(usages.keys()))

        # Truncate list to max number of included syllables
        truncate_point = np.where(syllables == truncate_syllable)[0]

        # Get syllable truncate index if it wasn't found
        if truncate_point is None or len(truncate_point) != 1:
            truncate_point = len(syllables)
        else:
            truncate_point = truncate_point[0]

        syllables = syllables[:truncate_point]

        usages = np.array(list(usages.values()), dtype='float')

        # Truncate syllable usages
        usages = usages[:truncate_point] + smoothing
        usages /= usages.sum()

        # Compute transition matrix from given syllable
        tm = get_transition_matrix([v],
                                   max_syllable=100,
                                   normalize='none',
                                   smoothing=0.0,

                                   disable_output=True)[0] + tm_smoothing

        # Truncate transition matrix
        tm = tm[:truncate_point, :truncate_point]

        # Normalize TM
        if normalize == 'bigram':
            tm /= tm.sum()
        elif normalize == 'rows':
            tm /= tm.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            tm /= tm.sum(axis=0, keepdims=True)

        # Compute entropy rate and append to syllable entropy rate
        ent.append(-np.sum(usages[:, None] * tm * np.log2(tm)))

    return ent
