"""Run all five missions in order (M1 -> M5)."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation, m5_report
from missions import m6_carbon_scheduling


def main():
    for m in (m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation):
        m.run(verbose=True)
        print()
    m5_report.run(verbose=True)
    print()
    m6_carbon_scheduling.run(verbose=True)  # Extension 5 (Your Turn, not graded by verify.py)


if __name__ == "__main__":
    main()
