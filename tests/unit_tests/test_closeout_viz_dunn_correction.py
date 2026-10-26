"""Focused current-upstream regression; generated arrays only."""
import numpy as np
import pandas as pd
import pytest


def test_dunn_multiple_comparisons_correct_across_syllables():
    from moseq2_viz.model.stat import compute_pvalues_for_group_pairs
    from statsmodels.stats.multitest import multipletests
    n = 1000
    raw = np.array([.001, .01, .04, .2])
    null = np.zeros((n, 4))
    for j, value in enumerate(raw):
        null[:int(value * n) - 1, j] = 2.
    actual, _ = compute_pvalues_for_group_pairs(
        {("a", "b"): np.ones(4)}, {("a", "b"): null},
        pd.DataFrame({"is_sig": [True] * 4}), ["a", "b"], n_perm=n)
    np.testing.assert_allclose(actual[("a", "b")], multipletests(raw, method="fdr_bh")[1])
