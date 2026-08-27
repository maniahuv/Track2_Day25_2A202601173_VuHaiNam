"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    # Extension 2: right-size memory-bound GPUs (low MBU) using $/GB-VRAM + measured
    # bandwidth need, instead of just picking the cheapest $/GPU-hr card.
    catalog_rows = catalog_by_type()
    catalog_num = {
        gtype: {**row, "on_demand_hr": num(row["on_demand_hr"]),
                "hbm_gb": num(row["hbm_gb"]), "peak_bw_tbs": num(row["peak_bw_tbs"])}
        for gtype, row in catalog_rows.items()
    }
    dollars_per_gb = {
        gtype: round(metrics.dollars_per_gb_vram(row["on_demand_hr"], row["hbm_gb"]), 4)
        for gtype, row in catalog_num.items()
    }

    HOURS_PER_MONTH = 720
    rightsizing = []
    rightsize_monthly_savings = 0.0

    achieved_bw_by_gpu = {}
    for r in tel:
        achieved_bw_by_gpu.setdefault(r["gpu_id"], []).append(num(r["achieved_bw_tbs"]))

    for s in summary:
        if not metrics.is_memory_bound_underutilized(s["mbu"]):
            continue
        avg_bw = sum(achieved_bw_by_gpu[s["gpu_id"]]) / len(achieved_bw_by_gpu[s["gpu_id"]])
        rec = metrics.recommend_rightsize(s["gpu_type"], avg_bw, catalog_num)
        if rec is None:
            continue
        monthly_savings = rec["savings_per_hr"] * HOURS_PER_MONTH
        rightsize_monthly_savings += monthly_savings
        rightsizing.append({**rec, "gpu_id": s["gpu_id"], "mbu": s["mbu"],
                             "monthly_savings": round(monthly_savings, 2)})

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print(f"\n-- Extension 2: right-sizing by MBU (target MBU >= {metrics.MBU_TARGET:.0%}) --")
        print(f"$/GB-VRAM by GPU type: " +
              ", ".join(f"{g}=${v:.4f}" for g, v in sorted(dollars_per_gb.items(), key=lambda x: x[1])))
        if rightsizing:
            print(f"{'GPU':14}{'MBU':>7}{'from':>7}{'to':>7}{'$/hr from':>11}{'$/hr to':>10}{'monthly $':>12}")
            for r in rightsizing:
                print(f"{r['gpu_id']:14}{r['mbu']:>7}{r['from_type']:>7}{r['to_type']:>7}"
                      f"{r['from_hr']:>11.2f}{r['to_hr']:>10.2f}{r['monthly_savings']:>12,.0f}")
            print(f"Right-sizing all memory-bound GPUs (assuming 24/7, {HOURS_PER_MONTH}h/month): "
                  f"${rightsize_monthly_savings:,.0f}/month saved")
        else:
            print("No GPU both (a) below MBU target and (b) has a cheaper catalog candidate "
                  "with enough bandwidth headroom.")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "dollars_per_gb_vram": dollars_per_gb, "rightsizing": rightsizing,
            "rightsize_monthly_savings": round(rightsize_monthly_savings, 2)}


if __name__ == "__main__":
    run()
