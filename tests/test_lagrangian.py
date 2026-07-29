from riskaware_saferrl.safety.lagrangian import LagrangeMultiplier


def test_multiplier_is_projected_to_non_negative_values() -> None:
    multiplier = LagrangeMultiplier(value=0.0, learning_rate=1.0)
    multiplier.update(observed_cost=0.0, cost_limit=1.0)
    assert multiplier.value == 0.0


def test_multiplier_increases_when_cost_exceeds_limit() -> None:
    multiplier = LagrangeMultiplier(value=0.0, learning_rate=0.5)
    multiplier.update(observed_cost=3.0, cost_limit=1.0)
    assert multiplier.value == 1.0
