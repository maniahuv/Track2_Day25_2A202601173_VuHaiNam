"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Extension 3: one-time write premium for caching a prefix (Anthropic-style ~1.25x
# the normal input price), used by cache_is_worth_it() to find the break-even.
CACHE_WRITE_MULT = 1.25

# Extension 4: routing rule — cap reasoning traffic at this share of *requests*.
# Reasoning is a per-request routing decision (does this query get sent to a
# reasoning model), so requests — not tokens — is the unit the cap controls.
REASONING_TRAFFIC_CAP = 0.10


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0

    # Extension 3: measure how many requests re-read each (team, project) cache
    # prefix — that's the natural "shared context" unit in this dataset (e.g. a
    # system prompt / retrieved doc reused across a project's requests).
    reads_by_prefix = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            reads_by_prefix[(r["team"], r["project"])] += 1
    avg_cache_reads = sum(reads_by_prefix.values()) / len(reads_by_prefix) if reads_by_prefix else 0.0

    worth_it_by_tier = {}
    for tier, (pin, _pout) in MODEL_PRICES.items():
        worth_it_by_tier[tier] = pricing.cache_is_worth_it(
            avg_cache_reads, write_cost_per_m=pin * CACHE_WRITE_MULT, price_in_per_m=pin,
        )

    # Extension 4: split $ and Wh by is_reasoning, using the same optimized
    # pricing path (cascade + batch + break-even-gated cache) so the reasoning
    # slice is measured on an apples-to-apples basis with the rest of traffic.
    reasoning_rows = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    reasoning_cost = reasoning_wh = 0.0
    normal_cost = normal_wh = 0.0

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = int(num(r["is_reasoning"])) == 1
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier) + batch API always apply; prompt caching
        # is only applied if cache_is_worth_it() clears break-even for this tier —
        # otherwise the write premium would outweigh the read discount.
        pin, pout = MODEL_PRICES[r["route_tier"]]
        use_cache = worth_it_by_tier[r["route_tier"]]
        req_cost = pricing.request_cost(
            inp, out, pin, pout, cached_in=(cached if use_cache else 0), batch=is_batch
        )
        opt_cost += req_cost

        req_wh = sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_cost += req_cost
            reasoning_wh += req_wh
        else:
            normal_cost += req_cost
            normal_wh += req_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension 4: reasoning traffic share (by request count, the routing unit)
    # vs. its share of optimized $ and Wh — these diverge because reasoning
    # requests run larger and cost 80x more energy per token than normal ones.
    reasoning_req_share = len(reasoning_rows) / len(rows) * 100 if rows else 0.0
    reasoning_cost_share = reasoning_cost / opt_cost * 100 if opt_cost else 0.0
    total_wh = reasoning_wh + normal_wh
    reasoning_wh_share = reasoning_wh / total_wh * 100 if total_wh else 0.0

    # Routing rule: if reasoning is already at/under a cap, there's nothing to
    # demote. Otherwise the highest-token reasoning requests (the priciest ones
    # to leave unmanaged) lose reasoning access first: downgraded to
    # route_tier="small" AND off the 80x energy multiplier. Both effects — cheaper
    # model tier *and* cheaper energy per query — are what a real router would
    # produce, priced the same way the rest of M2 prices requests (not a relabel).
    def simulate_cap(cap_frac: float) -> dict:
        target_n = int(len(rows) * cap_frac)
        demoted = []
        if len(reasoning_rows) > target_n:
            by_size_desc = sorted(reasoning_rows, key=lambda r: num(r["input_tokens"]) + num(r["output_tokens"]), reverse=True)
            demoted = by_size_desc[target_n:]  # keep the cap's worth on reasoning; demote the rest

        sav_usd = sav_wh = 0.0
        small_pin, small_pout = MODEL_PRICES["small"]
        for r in demoted:
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            cached = int(num(r["cached_input_tokens"]))
            is_batch = bool(int(num(r["is_batch"])))
            use_cache_before = worth_it_by_tier[r["route_tier"]]
            pin, pout = MODEL_PRICES[r["route_tier"]]
            cost_before = pricing.request_cost(inp, out, pin, pout, cached_in=(cached if use_cache_before else 0), batch=is_batch)

            use_cache_after = worth_it_by_tier["small"]
            cost_after = pricing.request_cost(inp, out, small_pin, small_pout, cached_in=(cached if use_cache_after else 0), batch=is_batch)

            sav_usd += cost_before - cost_after
            sav_wh += (
                sustainability.wh_per_query(inp + out, is_reasoning=True)
                - sustainability.wh_per_query(inp + out, is_reasoning=False)
            )
        return {"cap": cap_frac, "demoted": len(demoted), "savings_usd": sav_usd, "savings_wh": sav_wh}

    cap_at_target = simulate_cap(REASONING_TRAFFIC_CAP)
    # Current traffic (8.4%) is already under the 10% default cap, so also show a
    # tighter 5% cap to demonstrate the mechanism actually demoting requests.
    cap_tighter = simulate_cap(0.05)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print(f"\n-- Extension 3: cache_is_worth_it() --")
        print(f"cache prefixes tracked (team,project pairs): {len(reads_by_prefix)}")
        print(f"avg re-reads per prefix: {avg_cache_reads:.1f}")
        for tier, (pin, _pout) in MODEL_PRICES.items():
            be = pricing.cache_breakeven_reads(pin * CACHE_WRITE_MULT, pin)
            verdict = "WORTH IT" if worth_it_by_tier[tier] else "NOT worth it"
            print(f"  tier={tier:6} price_in=${pin:.2f}/1M  write=${pin*CACHE_WRITE_MULT:.3f}/1M  "
                  f"breakeven={be:.1f} reads  measured avg={avg_cache_reads:.1f} reads  -> {verdict}")

        print(f"\n-- Extension 4: reasoning budget --")
        print(f"reasoning requests: {len(reasoning_rows)}/{len(rows)}  ({reasoning_req_share:.1f}% of traffic by request)")
        print(f"reasoning $/day: ${reasoning_cost:.2f}  "
              f"({reasoning_cost_share:.1f}% of optimized daily $)")
        print(f"reasoning Wh/day: {reasoning_wh:,.1f} Wh  ({reasoning_wh_share:.1f}% of total Wh)  "
              f"vs non-reasoning: {normal_wh:,.1f} Wh")
        print(f"  -> reasoning is {reasoning_req_share:.1f}% of requests but {reasoning_wh_share:.1f}% of "
              f"energy, because each reasoning query already runs bigger AND burns "
              f"{sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x the energy per token.")
        print(f"routing rule: keep only the smallest reasoning requests up to a traffic "
              f"cap; demote the largest excess ones to route_tier=small (loses reasoning "
              f"pricing tier AND the {sustainability.REASONING_ENERGY_MULTIPLIER:.0f}x energy multiplier)")
        for cap_result in (cap_at_target, cap_tighter):
            cap_pct = cap_result["cap"]
            if cap_result["demoted"] == 0:
                print(f"  cap={cap_pct:.0%}: reasoning ({reasoning_req_share:.1f}%) already at/under "
                      f"cap -> nothing to demote")
            else:
                print(f"  cap={cap_pct:.0%}: demote {cap_result['demoted']} largest reasoning requests -> "
                      f"saves ${cap_result['savings_usd']:.2f}/day and {cap_result['savings_wh']:,.1f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "avg_cache_reads": round(avg_cache_reads, 2), "cache_worth_it_by_tier": worth_it_by_tier,
        "reasoning_requests": len(reasoning_rows), "reasoning_req_share_pct": round(reasoning_req_share, 1),
        "reasoning_daily_usd": round(reasoning_cost, 2), "reasoning_cost_share_pct": round(reasoning_cost_share, 1),
        "reasoning_daily_wh": round(reasoning_wh, 1), "reasoning_wh_share_pct": round(reasoning_wh_share, 1),
        "normal_daily_wh": round(normal_wh, 1),
        "reasoning_cap_10pct": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cap_at_target.items()},
        "reasoning_cap_5pct": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cap_tighter.items()},
    }


if __name__ == "__main__":
    run()
