"""Extension 5 — tests for missions/m6_carbon_scheduling.py.

New file, does not modify the graded test files.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import generate
from missions import m6_carbon_scheduling as m6


def setup_module(_module):
    generate.main()  # deterministic (seed=25) synthetic data


def test_only_interruptible_jobs_are_considered_movable():
    from missions._common import load_csv, num
    jobs = load_csv("workloads.csv")
    interruptible_ids = {j["job_id"] for j in jobs if bool(int(num(j["interruptible"])))}
    r = m6.run(verbose=False)
    moved_ids = {m["job_id"] for m in r["job_moves"]}
    assert moved_ids == interruptible_ids
    assert len(moved_ids) < len(jobs)  # sanity: not every job in the dataset is interruptible


def test_cleanest_region_matches_sustainability_table():
    from finops import sustainability
    r = m6.run(verbose=False)
    assert r["cleanest_region"] == min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    assert r["cheapest_region"] == min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)


def test_moving_to_cleanest_region_never_increases_carbon():
    r = m6.run(verbose=False)
    for m in r["job_moves"]:
        assert m["carbon_clean_g"] <= m["carbon_current_g"]
        assert m["carbon_saved_g"] >= 0


def test_total_carbon_saved_is_positive_and_sums_correctly():
    r = m6.run(verbose=False)
    manual_total_kg = sum(m["carbon_saved_g"] for m in r["job_moves"]) / 1000
    assert r["total_carbon_saved_kg"] > 0
    assert abs(r["total_carbon_saved_kg"] - manual_total_kg) < 0.1  # rounding tolerance


def test_region_table_covers_all_five_regions():
    r = m6.run(verbose=False)
    assert len(r["region_table"]) == 5
    regions = {row["region"] for row in r["region_table"]}
    assert regions == {"us-east-1", "us-west-2", "europe-north1", "europe-central2", "us-east-wa"}
