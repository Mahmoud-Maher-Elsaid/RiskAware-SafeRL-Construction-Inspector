from riskaware_saferrl.domain_randomization import DomainRandomizer


def test_seed_reproduces_domain_sequence() -> None:
    first = [DomainRandomizer(42).sample().to_dict() for _ in range(2)]
    second = [DomainRandomizer(42).sample().to_dict() for _ in range(2)]
    assert first == second


def test_reference_domain_is_not_randomized() -> None:
    reference = DomainRandomizer(42, enabled=False).sample()
    assert reference.lighting_intensity == 1.0
    assert reference.sensor_noise_std == 0.0
