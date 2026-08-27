"""Extension 4 — tests for the reasoning-budget split in m2_inference_levers.run().

New file, does not modify the graded test files.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import generate
from missions import m2_inference_levers as m2


def setup_module(_module):
    generate.main()  # deterministic (seed=25) synthetic data


def test_reasoning_share_reported_and_energy_disproportionate():
    r = m2.run(verbose=False)
    assert 0 <= r["reasoning_req_share_pct"] <= 100
    assert 0 <= r["reasoning_wh_share_pct"] <= 100
    # Reasoning multiplies energy 80x per token, so its energy share should be
    # far larger than its share of requests on any realistic traffic mix.
    assert r["reasoning_wh_share_pct"] > r["reasoning_req_share_pct"]


def test_reasoning_daily_wh_matches_normal_plus_reasoning_total():
    r = m2.run(verbose=False)
    assert r["reasoning_daily_wh"] > 0
    assert r["normal_daily_wh"] > 0


def test_tighter_cap_demotes_more_than_default_cap():
    r = m2.run(verbose=False)
    # A 5% cap is stricter than the 10% default, so it can only demote the same
    # or more requests, never fewer.
    assert r["reasoning_cap_5pct"]["demoted"] >= r["reasoning_cap_10pct"]["demoted"]


def test_demoting_never_saves_negative_energy():
    r = m2.run(verbose=False)
    for cap_result in (r["reasoning_cap_10pct"], r["reasoning_cap_5pct"]):
        assert cap_result["savings_wh"] >= 0
        assert cap_result["savings_usd"] >= 0
