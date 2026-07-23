from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.safety import SafetyShield


def test_shield_replaces_known_unsafe_action() -> None:
    base_env = ConstructionInspectionEnv(
        size=6, n_obstacles=1, n_hazards=1, n_workers=1, n_restricted=1
    )
    env = SafetyShield(base_env)
    env.reset(seed=1)

    unsafe_action = None
    for action in range(4):
        if not base_env.is_action_safe(action):
            unsafe_action = action
            break

    if unsafe_action is None:
        return

    _, _, _, _, info = env.step(unsafe_action)
    assert info["shield_active"] is True
    assert info["executed_action"] != unsafe_action
