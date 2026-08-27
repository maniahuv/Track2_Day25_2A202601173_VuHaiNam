"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}

# Extension 3: one-time write premium for caching a prefix (Anthropic-style ~1.25x
# the normal input price), used by cache_is_worth_it() to find the break-even.
CACHE_WRITE_MULT = 1.25


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

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier) + batch API always apply; prompt caching
        # is only applied if cache_is_worth_it() clears break-even for this tier —
        # otherwise the write premium would outweigh the read discount.
        pin, pout = MODEL_PRICES[r["route_tier"]]
        use_cache = worth_it_by_tier[r["route_tier"]]
        opt_cost += pricing.request_cost(
            inp, out, pin, pout, cached_in=(cached if use_cache else 0), batch=is_batch
        )

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

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

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "avg_cache_reads": round(avg_cache_reads, 2), "cache_worth_it_by_tier": worth_it_by_tier,
    }


if __name__ == "__main__":
    run()
