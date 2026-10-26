"""Focused upstream regression; synthetic fixtures only."""
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import pytest
import pandas as pd


def test_metadata_columns_follow_requested_key_order(monkeypatch):
    import moseq2_viz.scalars.util as util
    def values(filename, path=None):
        if path == "scalars":
            return {"centroid_x_px": np.ones(4), "centroid_y_px": np.ones(4)}
        raise KeyError("synthetic fixture has no ROI")
    monkeypatch.setattr(util, "h5_to_dict", values)
    monkeypatch.setattr(util, "get_timestamps_from_h5", lambda path: np.arange(4))
    keys = ["SessionName", "SubjectName", "StartTime"]
    index = {"files": {"synthetic": {"path": ["synthetic.h5", "synthetic.yaml"],
                                     "group": "synthetic",
                                     "metadata": {key: "synthetic" for key in keys[::-1]}}}}
    frame = util.scalars_to_dataframe(index, include_keys=keys, force_conversion=False)
    assert [key for key in frame.columns if key in keys] == keys
