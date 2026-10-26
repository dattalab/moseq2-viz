"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_transition_matrix_counts_initial_transition():
    from moseq2_viz.model.trans_graph import get_transition_matrix
    actual = get_transition_matrix([np.array([0, 0, 1, 1, 0, 0])], max_syllable=2,
                                   normalize=None, combine=True, smoothing=0)
    np.testing.assert_array_equal(actual, [[0, 1], [1, 0]])
