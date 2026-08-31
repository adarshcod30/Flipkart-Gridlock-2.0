import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridlock.features import (
    PROFILE_COLUMNS,
    add_spatiotemporal_features,
    attach_day48_profile,
    build_day48_profile,
    decode_geohash,
)


def test_decode_geohash_is_within_bounds():
    lat, lon = decode_geohash("qp02z1")
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0


def test_decode_geohash_is_deterministic():
    assert decode_geohash("qp02z1") == decode_geohash("qp02z1")


def test_decode_geohash_distinguishes_locations():
    assert decode_geohash("qp02z1") != decode_geohash("9q8yy0")


def test_add_spatiotemporal_features_time_idx():
    df = pd.DataFrame({"geohash": ["qp02z1"], "timestamp": ["14:30"]})
    out = add_spatiotemporal_features(df)
    assert out.loc[0, "hour"] == 14
    assert out.loc[0, "minute"] == 30
    assert out.loc[0, "time_idx"] == 14 * 4 + 30 // 15


def test_build_day48_profile_has_no_missing_values():
    df = pd.DataFrame(
        {
            "geohash": ["gh1", "gh1", "gh2"],
            "day": [48, 48, 48],
            "time_idx": [0, 1, 0],
            "demand": [0.1, 0.2, 0.3],
        }
    )
    profile, global_profile = build_day48_profile(df)
    assert list(profile.columns) == PROFILE_COLUMNS
    assert not profile.isna().any().any()
    assert not global_profile.isna().any()


def test_attach_day48_profile_fills_unseen_geohash():
    df = pd.DataFrame(
        {
            "geohash": ["gh1", "gh1"],
            "day": [48, 48],
            "time_idx": [0, 1],
            "demand": [0.1, 0.2],
        }
    )
    profile, global_profile = build_day48_profile(df)

    test_rows = pd.DataFrame({"geohash": ["gh_never_seen"], "time_idx": [0]})
    attached = attach_day48_profile(test_rows, profile, global_profile)
    assert not attached[PROFILE_COLUMNS].isna().any().any()
    assert attached.loc[0, "d48_t0"] == pytest.approx(global_profile["d48_t0"])
