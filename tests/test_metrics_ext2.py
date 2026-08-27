"""Extension 2 — tests for MBU-based right-sizing (finops.metrics).

New file, does not modify the graded test_metrics.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import metrics

CATALOG = {
    "H100": {"on_demand_hr": 2.5, "hbm_gb": 80, "peak_bw_tbs": 3.35},
    "A100": {"on_demand_hr": 1.79, "hbm_gb": 80, "peak_bw_tbs": 2.0},
    "A10G": {"on_demand_hr": 1.0, "hbm_gb": 24, "peak_bw_tbs": 0.6},
    "L4":   {"on_demand_hr": 0.8, "hbm_gb": 24, "peak_bw_tbs": 0.3},
}


def test_dollars_per_gb_vram():
    assert abs(metrics.dollars_per_gb_vram(2.5, 80) - 2.5 / 80) < 1e-9
    assert metrics.dollars_per_gb_vram(1.0, 0) == float("inf")


def test_is_memory_bound_underutilized():
    assert metrics.is_memory_bound_underutilized(0.3) is True
    assert metrics.is_memory_bound_underutilized(0.8) is False
    assert metrics.is_memory_bound_underutilized(0.6, target=0.60) is False


def test_recommend_rightsize_finds_cheaper_candidate_with_headroom():
    # H100 achieving only 0.5 TB/s (needs ~0.575 with headroom): the cheapest
    # catalog card that still covers it is A10G (peak 0.6), not just any GPU.
    rec = metrics.recommend_rightsize("H100", achieved_bw_tbs=0.5, catalog=CATALOG)
    assert rec is not None
    assert rec["to_type"] == "A10G"
    assert rec["savings_per_hr"] > 0
    assert rec["savings_pct"] > 0


def test_recommend_rightsize_picks_cheapest_qualifying_not_just_any():
    # H100 achieving 1.5 TB/s (needs ~1.725): A10G/L4 fall short, only A100 (2.0)
    # qualifies among the cheaper options.
    rec = metrics.recommend_rightsize("H100", achieved_bw_tbs=1.5, catalog=CATALOG)
    assert rec is not None
    assert rec["to_type"] == "A100"


def test_recommend_rightsize_rejects_candidate_without_bandwidth_headroom():
    # A100 achieving 0.55 TB/s * 1.15 headroom ~= 0.63 TB/s: A10G (0.6) and L4 (0.3)
    # both fall short, so no downgrade should be proposed even though they're cheaper.
    rec = metrics.recommend_rightsize("A100", achieved_bw_tbs=0.55, catalog=CATALOG)
    assert rec is None


def test_recommend_rightsize_returns_none_for_unknown_gpu():
    assert metrics.recommend_rightsize("UNKNOWN", 1.0, CATALOG) is None
