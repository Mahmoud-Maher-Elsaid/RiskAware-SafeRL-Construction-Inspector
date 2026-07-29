from scripts.run_stage9_ablations import VARIANTS


def test_required_ablation_variants_have_exact_differences() -> None:
    assert len(VARIANTS) == 6
    assert {variant["name"] for variant in VARIANTS} == {
        "riskshield_ppo_full",
        "without_predictive_safety_shield",
        "without_cv_risk_injection",
        "without_action_masking",
        "without_constrained_cost_objective",
        "one_step_shield",
    }
    assert all(variant["difference"] for variant in VARIANTS)
    assert (
        next(variant for variant in VARIANTS if variant["name"] == "one_step_shield")[
            "shield_horizon"
        ]
        == 1
    )
