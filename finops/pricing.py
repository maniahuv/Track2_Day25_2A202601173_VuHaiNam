"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# Extension 1 — per-GPU-type spot interruption rates (neocloud premium SKUs are
# less contended than commodity inference cards, so they get reclaimed less often).
GPU_INTERRUPT_RATE = {
    "H100": 0.03,
    "H200": 0.03,
    "B200": 0.02,
    "MI300X": 0.04,
    "A100": 0.05,
    "A10G": 0.09,
    "L4": 0.10,
}
DEFAULT_INTERRUPT_RATE = 0.05


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str | None = None,
    job_days: float | None = None,
    reserved_discount_1yr: float = 0.20,
    reserved_discount_3yr: float = 0.45,
    max_viable_interrupt_rate: float = 0.12,
) -> str:
    """Tier policy extended with GPU-specific interruption rate + 1yr-vs-3yr duration fit.

    Adds two factors the v1 policy ignores:
      1. Interruption rate varies a lot by GPU type (commodity inference cards like
         A10G/L4 get reclaimed far more often than H100/B200). A job on a
         high-churn GPU may not actually be spot-viable even if it is flagged
         `interruptible`, so we fall through to duty-cycle logic instead.
      2. A commitment should match how long the job actually runs. `job_days` here
         is the length of the *observed run*, not a promised commitment term — a
         short, bursty training job (a few days, then it's done) shouldn't be
         reserved even at high duty cycle, while a steady non-interruptible
         service observed running for weeks is exactly the profile reserved
         pricing is meant for. Jobs that clear the bar compare 1yr vs 3yr and
         take whichever break-even is actually cleared.
    """
    duty = max(0.0, hours_per_day) / 24.0
    interrupt_rate = GPU_INTERRUPT_RATE.get((gpu_type or "").upper(), DEFAULT_INTERRUPT_RATE)

    if interruptible and hours_per_day < 24 and interrupt_rate <= max_viable_interrupt_rate:
        return "spot"

    be_1yr = break_even_utilization(reserved_discount_1yr)
    be_3yr = break_even_utilization(reserved_discount_3yr)

    # A handful of short observed days (bursty training run) doesn't look like a
    # job worth committing hardware to, even if its duty cycle is momentarily high.
    commits_long_enough = job_days is None or job_days >= 14

    if commits_long_enough and duty >= be_3yr:
        return "reserved_3yr"
    if commits_long_enough and duty >= be_1yr:
        return "reserved_1yr"
    return "on_demand"


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    price_in_per_m: float,
    read_discount: float = 0.10,   # 10% = 90% off, same knob as request_cost's cache_discount
) -> bool:
    """Does prompt caching actually save money for this traffic pattern?

    Caching a prefix isn't free: writing it costs `write_cost_per_m` $/1M tokens
    (a one-time premium over the normal input price, e.g. Anthropic's ~1.25x
    write surcharge). Reading it back is cheap (`read_discount` x normal price).
    The write only pays for itself once enough *re-reads* have accumulated to
    beat what those tokens would have cost at the normal input price every time.

    Break-even: write_cost <= avg_cache_reads * (price_in_per_m - price_in_per_m*read_discount)
             -> avg_cache_reads >= write_cost / (price_in_per_m * (1 - read_discount))

    Returns True iff the measured average re-read count clears that bar.
    """
    if price_in_per_m <= 0 or read_discount >= 1.0:
        return False
    per_read_saving = price_in_per_m * (1.0 - read_discount)
    if per_read_saving <= 0:
        return False
    breakeven_reads = write_cost_per_m / per_read_saving
    return avg_cache_reads >= breakeven_reads


def cache_breakeven_reads(write_cost_per_m: float, price_in_per_m: float, read_discount: float = 0.10) -> float:
    """Minimum average re-reads per cached prefix needed for caching to pay off."""
    per_read_saving = price_in_per_m * (1.0 - read_discount)
    if per_read_saving <= 0:
        return float("inf")
    return write_cost_per_m / per_read_saving


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
