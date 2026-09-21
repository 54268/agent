"""Predeclared Stage-9 G0 autonomy gate; no final Unknown metrics here."""
from __future__ import annotations


def evaluate_g0(pair: dict, cfg: dict) -> dict:
    per_agent = {}
    for role in ("identity", "impairment"):
        left = pair["datasets"]["oracle"]["agents"][role]
        right = pair["datasets"]["wisig"]["agents"][role]
        # Compare distributions over mutually exclusive *actions*. Tool
        # inclusion rates can sum above one for Top-2 and are not a TVD.
        oracle_actions = left["known"]["action_counts"]
        wisig_actions = right["known"]["action_counts"]
        oracle_n = max(left["known"]["samples"], 1)
        wisig_n = max(right["known"]["samples"], 1)
        tvd = 0.5 * sum(abs(
            oracle_actions[name] / oracle_n - wisig_actions[name] / wisig_n)
            for name in oracle_actions)
        checks = {
            "nonconstant_oracle": left["known"]["distinct_actions"] >= 2
                and left["known"]["majority_action_fraction"]
                <= float(cfg["g0_max_majority_fraction"]),
            "nonconstant_wisig": right["known"]["distinct_actions"] >= 2
                and right["known"]["majority_action_fraction"]
                <= float(cfg["g0_max_majority_fraction"]),
            "gain_over_random_oracle": left["known"]["gain_vs_random"]
                > float(cfg["g0_min_gain"]),
            "gain_over_random_wisig": right["known"]["gain_vs_random"]
                > float(cfg["g0_min_gain"]),
            "gain_over_fixed_oracle": left["known"]["gain_vs_fixed"]
                > float(cfg["g0_min_gain"]),
            "gain_over_fixed_wisig": right["known"]["gain_vs_fixed"]
                > float(cfg["g0_min_gain"]),
            "cross_dataset_usage_difference": tvd
                >= float(cfg["g0_min_usage_tvd"]),
        }
        per_agent[role] = {
            "checks": checks, "passed": all(checks.values()),
            "tool_usage_tvd": tvd,
        }
    return {
        "passed": all(value["passed"] for value in per_agent.values()),
        "agents": per_agent,
        "next_step": ("implement Open-Set Examiner and test G1 complementarity"
                      if all(value["passed"] for value in per_agent.values())
                      else "stop at G0; redesign local tools/policy before communication"),
    }
