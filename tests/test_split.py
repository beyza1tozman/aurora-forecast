"""Time-based split: ordered, non-overlapping, with an embargo gap."""

import pandas as pd

from aurora.dataset import EMBARGO, HORIZONS, assign_split, target_start

INDEX = pd.date_range("1998-01-01", "2026-09-13", freq="1h", tz="UTC")
ORDER = ["train", "val", "test"]


def test_every_row_has_exactly_one_label():
    labels = assign_split(INDEX)
    assert labels.notna().all()
    assert set(labels) == {"train", "val", "test", "embargo"}


def test_splits_are_ordered_and_separated_by_embargo():
    labels = assign_split(INDEX)
    for earlier, later in zip(ORDER, ORDER[1:], strict=False):
        last = labels.index[labels == earlier].max()
        first = labels.index[labels == later].min()
        assert first - last >= EMBARGO


def test_targets_of_a_split_end_before_the_next_split_starts():
    labels = assign_split(INDEX)
    for earlier, later in zip(ORDER, ORDER[1:], strict=False):
        rows = labels.index[labels == earlier]
        latest_target_end = target_start(rows, max(HORIZONS)).max() + pd.Timedelta("3h")
        assert latest_target_end < labels.index[labels == later].min()


def test_split_boundaries():
    labels = assign_split(INDEX)
    assert labels[pd.Timestamp("2014-12-31 23:00", tz="UTC")] == "train"
    assert labels["2015-01-15"].eq("embargo").all()
    assert labels["2015-02-01"].eq("val").all()
    assert labels["2020-01-10"].eq("embargo").all()
    assert labels["2024-05-10"].eq("test").all()
