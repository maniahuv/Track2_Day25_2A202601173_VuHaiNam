"""Extension 5 — Carbon-aware Scheduling (Rubric D.5).

For every interruptible job in workloads.csv, compare running it in the default
region (us-east-1) against every region in finops.sustainability.REGION_CARBON:
energy (Wh), carbon (gCO2e), and electricity cost ($) if it moved. Interruptible
jobs are the ones actually free to be rescheduled to wherever the grid is
cleanest/cheapest at the moment — a non-interruptible 24/7 inference service
can't just hop regions without a real migration.

Run: python missions/m6_carbon_scheduling.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

DAYS = 30
DEFAULT_REGION = "us-east-1"


def _job_energy_wh(job: dict, catalog: dict) -> float:
    """Total energy for a job's full run: watts * gpu_hours."""
    gtype = job["gpu_type"]
    watts = num(catalog[gtype]["watts"])
    hpd = num(job["hours_per_day"])
    days = num(job["days"])
    ngpu = int(num(job["num_gpus"]))
    gpu_hours = hpd * days * ngpu
    return watts * gpu_hours  # Wh, since watts * hours = Wh


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    interruptible_jobs = [j for j in jobs if bool(int(num(j["interruptible"])))]

    cleanest_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    cheapest_region = min(sustainability.REGION_PRICE_KWH, key=sustainability.REGION_PRICE_KWH.get)

    # "Balanced" region: normalize both carbon and price to 0-1 and minimize the
    # sum, so a region has to be reasonably good on both axes, not just win one.
    carbon_vals = sustainability.REGION_CARBON
    price_vals = sustainability.REGION_PRICE_KWH
    c_min, c_max = min(carbon_vals.values()), max(carbon_vals.values())
    p_min, p_max = min(price_vals.values()), max(price_vals.values())

    def _norm(v, lo, hi):
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    balanced_region = min(
        carbon_vals,
        key=lambda r: _norm(carbon_vals[r], c_min, c_max) + _norm(price_vals[r], p_min, p_max),
    )

    job_moves = []
    total_carbon_saved_g = 0.0
    total_dollar_delta = 0.0
    for j in interruptible_jobs:
        wh = _job_energy_wh(j, cat)
        current_carbon = sustainability.carbon_g(wh, DEFAULT_REGION)
        current_cost = sustainability.energy_cost_usd(wh, DEFAULT_REGION)
        clean_carbon = sustainability.carbon_g(wh, cleanest_region)
        clean_cost = sustainability.energy_cost_usd(wh, cleanest_region)

        carbon_saved = current_carbon - clean_carbon
        carbon_saved_pct = carbon_saved / current_carbon * 100 if current_carbon else 0.0
        cost_delta = clean_cost - current_cost  # positive = moving costs more $

        total_carbon_saved_g += carbon_saved
        total_dollar_delta += cost_delta

        job_moves.append({
            "job_id": j["job_id"], "gpu_type": j["gpu_type"], "wh": round(wh, 1),
            "carbon_current_g": round(current_carbon, 1), "carbon_clean_g": round(clean_carbon, 1),
            "carbon_saved_g": round(carbon_saved, 1), "carbon_saved_pct": round(carbon_saved_pct, 1),
            "energy_cost_current_usd": round(current_cost, 2), "energy_cost_clean_usd": round(clean_cost, 2),
            "energy_cost_delta_usd": round(cost_delta, 2),
        })

    region_table = []
    for region in sustainability.REGION_CARBON:
        region_table.append({
            "region": region,
            "gco2_per_kwh": sustainability.REGION_CARBON[region],
            "usd_per_kwh": sustainability.REGION_PRICE_KWH[region],
        })
    region_table.sort(key=lambda r: r["gco2_per_kwh"])

    if verbose:
        print("== M6 Carbon-aware Scheduling (Extension 5) ==")
        print(f"{'region':16}{'gCO2/kWh':>11}{'$/kWh':>9}")
        for r in region_table:
            tags = []
            if r["region"] == cleanest_region:
                tags.append("cleanest")
            if r["region"] == cheapest_region:
                tags.append("cheapest")
            if r["region"] == balanced_region:
                tags.append("balanced")
            tag_str = f"  <- {', '.join(tags)}" if tags else ""
            print(f"{r['region']:16}{r['gco2_per_kwh']:>11}{r['usd_per_kwh']:>9.3f}{tag_str}")

        print(f"\nCleanest region : {cleanest_region} ({sustainability.REGION_CARBON[cleanest_region]} gCO2/kWh)")
        print(f"Cheapest region : {cheapest_region} (${sustainability.REGION_PRICE_KWH[cheapest_region]}/kWh)")
        print(f"Balanced region : {balanced_region} (best combined rank on both axes)")

        print(f"\n-- Moving each interruptible job from {DEFAULT_REGION} -> {cleanest_region} --")
        print(f"{'job':18}{'gpu':7}{'Wh':>10}{'carbon now':>12}{'carbon clean':>13}{'saved':>9}{'saved%':>8}{'$ delta':>9}")
        for m in job_moves:
            print(f"{m['job_id']:18}{m['gpu_type']:7}{m['wh']:>10,.0f}{m['carbon_current_g']:>12,.0f}"
                  f"{m['carbon_clean_g']:>13,.0f}{m['carbon_saved_g']:>9,.0f}{m['carbon_saved_pct']:>7.1f}%"
                  f"{m['energy_cost_delta_usd']:>+9.2f}")
        print(f"\nTotal carbon saved if all interruptible jobs move to {cleanest_region}: "
              f"{total_carbon_saved_g/1000:,.2f} kgCO2e")
        print(f"Total electricity cost delta: {'+' if total_dollar_delta>=0 else ''}${total_dollar_delta:,.2f} "
              f"({'more' if total_dollar_delta>=0 else 'less'} expensive than {DEFAULT_REGION})")
        print(
            "\nTrade-off note: europe-north1 is cleanest+3rd-cheapest here, so this dataset's "
            "carbon-optimal move happens to also cut $ — that won't always hold. The real cost "
            "of moving is latency: shipping interruptible batch/training jobs to a distant region "
            "is fine (no user waiting), but the same move for a live inference workload would add "
            "cross-region round-trip latency, which is why only interruptible=1 jobs are considered "
            "movable here."
        )

    return {
        "cleanest_region": cleanest_region, "cheapest_region": cheapest_region,
        "balanced_region": balanced_region, "region_table": region_table,
        "job_moves": job_moves, "total_carbon_saved_kg": round(total_carbon_saved_g / 1000, 2),
        "total_energy_cost_delta_usd": round(total_dollar_delta, 2),
    }


if __name__ == "__main__":
    run()
