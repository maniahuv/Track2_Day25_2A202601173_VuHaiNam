"""Extension 3 — tests for cache_is_worth_it() / cache_breakeven_reads() (finops.pricing).

New file, does not modify the graded test_pricing.py.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing


def test_breakeven_reads_matches_manual_math():
    # write=$0.25/1M, price_in=$0.20/1M, read_discount=0.10 -> per-read saving = 0.18/1M
    # breakeven = 0.25 / 0.18 = 1.388...
    be = pricing.cache_breakeven_reads(write_cost_per_m=0.25, price_in_per_m=0.20)
    assert abs(be - (0.25 / 0.18)) < 1e-9


def test_worth_it_above_breakeven():
    # 300 re-reads vastly clears a ~1.4-read breakeven.
    assert pricing.cache_is_worth_it(avg_cache_reads=300, write_cost_per_m=0.25, price_in_per_m=0.20) is True


def test_not_worth_it_below_breakeven():
    # A prefix re-read only once never earns back a 25%-premium write.
    assert pricing.cache_is_worth_it(avg_cache_reads=1, write_cost_per_m=0.25, price_in_per_m=0.20) is False


def test_worth_it_exactly_at_breakeven_boundary():
    be = pricing.cache_breakeven_reads(write_cost_per_m=0.25, price_in_per_m=0.20)
    assert pricing.cache_is_worth_it(avg_cache_reads=be, write_cost_per_m=0.25, price_in_per_m=0.20) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=be - 0.01, write_cost_per_m=0.25, price_in_per_m=0.20) is False


def test_degenerate_inputs_are_safe():
    assert pricing.cache_is_worth_it(avg_cache_reads=10, write_cost_per_m=1.0, price_in_per_m=0.0) is False
    assert pricing.cache_breakeven_reads(write_cost_per_m=1.0, price_in_per_m=0.0) == float("inf")
