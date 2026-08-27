"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _build_analysis(r1: dict, levers: dict, baseline: float, sust: dict) -> str:
    """Human-readable root-cause explanation + prioritized recommendations.

    Regenerated every run from the actual mission results (lie GPUs, lever
    ranking, sustainability numbers) so it never drifts from the numbers above
    it, but the *reasoning* — why GPU-Util misleads, why purchasing dominates,
    what to do first — is written prose, not a template of the raw output.
    """
    lie_ids = [l["gpu_id"] for l in r1["lies"]]
    lie_detail = "; ".join(
        f"{l['gpu_id']} (util {l['gpu_util_pct']:.0f}%, MFU {l['mfu']:.2f})" for l in r1["lies"]
    )
    ranked = sorted(levers.items(), key=lambda kv: kv[1], reverse=True)
    top_lever, top_amount = ranked[0]
    top_pct = top_amount / sum(levers.values()) * 100 if sum(levers.values()) else 0.0

    return (
        f"**Why GPU-Util is a lie here:** `nvidia-smi`'s Util% only answers "
        f"\"is a kernel running right now\", not \"is the Tensor Core doing useful "
        f"work\". {lie_detail} read as fully busy while spending most of that time "
        f"stalled on HBM reads or waiting on kernel-launch overhead — the compute "
        f"units sit idle even though the clock reads active. Financially, paying "
        f"full on-demand rate for {', '.join(lie_ids)} while receiving under 30% of "
        f"the FLOPs means the majority of those GPU-hours bought no tokens at all.\n\n"
        f"**Priority order (highest ROI first):**\n"
        f"1. **{top_lever}** (${top_amount:,.0f}/mo, {top_pct:.0f}% of total savings) — "
        f"the largest lever because it acts on the *entire* GPU-hour base rather than "
        f"only the inference slice; a pure purchasing/contract change, no code required.\n"
        f"2. **Right-size util-lie GPUs** — downgrading {', '.join(lie_ids)} converts "
        f"wasted memory-bound H100/A10G capacity into cheaper hardware that fits the "
        f"actual bandwidth need, without touching application code.\n"
        f"3. **Inference levers (cascade/cache/batch)** — smaller in absolute $ today "
        f"because baseline inference spend is small relative to the purchasing base, "
        f"but this lever scales with traffic and should be built into the serving path "
        f"before volume grows.\n\n"
        f"**Sustainability tie-in:** region selection is not just a carbon story — "
        f"{sust.get('best_region', 'n/a')} is also cost-competitive on $/kWh here, so "
        f"moving interruptible workloads there cuts electricity spend and grid carbon "
        f"together. Reasoning traffic is the outlier: it is "
        f"{sust.get('reasoning_req_share_pct', 0):.1f}% of requests but "
        f"{sust.get('reasoning_wh_share_pct', 0):.1f}% of energy, so it is a carbon "
        f"lever almost independent of the dollar levers above — worth capping on its "
        f"own routing rule rather than folded into general cost optimization."
    )


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        # Extension 4: reasoning traffic is a disproportionate energy driver —
        # surfaced here alongside the rest of the sustainability numbers.
        "reasoning_req_share_pct": r2["reasoning_req_share_pct"],
        "reasoning_wh_share_pct": r2["reasoning_wh_share_pct"],
        "reasoning_cap_5pct_wh_saved_daily": r2["reasoning_cap_5pct"]["savings_wh"],
    }

    analysis = _build_analysis(r1, levers, baseline, sust)
    md = report.build_report(baseline, optimized, levers, sustainability=sust, analysis=analysis)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
