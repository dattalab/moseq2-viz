"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd


def test_default_model_merge_preserves_state_parameter_alignment(monkeypatch):
    from copy import deepcopy
    import moseq2_viz.model.util as util
    parameters = np.array([[[1., 0.]], [[3., 0.]], [[8., 0.]]])
    template = {"labels": {"synthetic": np.array([0, 1, 2])},
                "model_parameters": {"ar_mat": parameters.copy(), "kappa": "synthetic"}}
    candidate = {"labels": {"synthetic": np.array([2, 0, 1])},
                 "model_parameters": {"ar_mat": parameters[[1, 2, 0]].copy(), "kappa": "synthetic"}}
    monkeypatch.setattr(util.glob, "glob", lambda *args: ["synthetic-a.p", "synthetic-b.p"])
    monkeypatch.setattr(util, "parse_model_results", lambda path, **kw:
                        deepcopy(template if path == "synthetic-a.p" else candidate))
    result = util.merge_models("synthetic-unused", force_merge=True)
    np.testing.assert_array_equal(result["synthetic-b.p"]["labels"]["synthetic"],
                                  template["labels"]["synthetic"])
    np.testing.assert_array_equal(result["synthetic-b.p"]["model_parameters"]["ar_mat"], parameters)
