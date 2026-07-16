import torch

from riskaware_saferrl.envs import ConstructionInspectionEnv
from riskaware_saferrl.models import SemanticMapExtractor


def test_semantic_extractor_output_shape() -> None:
    env = ConstructionInspectionEnv()
    observation, _ = env.reset(seed=10)

    extractor = SemanticMapExtractor(
        env.observation_space,
        features_dim=256,
    )

    tensor_observation = {
        key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()
    }

    output = extractor(tensor_observation)

    assert output.shape == (1, 256)


def test_semantic_extractor_output_is_finite() -> None:
    env = ConstructionInspectionEnv()
    observation, _ = env.reset(seed=20)

    extractor = SemanticMapExtractor(
        env.observation_space,
        features_dim=128,
    )

    tensor_observation = {
        key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()
    }

    output = extractor(tensor_observation)

    assert output.shape == (1, 128)
    assert torch.isfinite(output).all()
