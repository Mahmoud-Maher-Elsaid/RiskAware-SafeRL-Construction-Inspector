import pytest

from riskaware_saferrl.callbacks.curriculum_progress import (
    curriculum_stage_for_update,
)


def test_curriculum_stage_boundaries() -> None:
    assert (
        curriculum_stage_for_update(
            0,
            easy_updates=25,
            medium_updates=25,
        )
        == "easy"
    )
    assert (
        curriculum_stage_for_update(
            24,
            easy_updates=25,
            medium_updates=25,
        )
        == "easy"
    )
    assert (
        curriculum_stage_for_update(
            25,
            easy_updates=25,
            medium_updates=25,
        )
        == "medium"
    )
    assert (
        curriculum_stage_for_update(
            49,
            easy_updates=25,
            medium_updates=25,
        )
        == "medium"
    )
    assert (
        curriculum_stage_for_update(
            50,
            easy_updates=25,
            medium_updates=25,
        )
        == "full"
    )


def test_curriculum_stage_rejects_invalid_schedule() -> None:
    with pytest.raises(ValueError, match="easy_updates"):
        curriculum_stage_for_update(
            0,
            easy_updates=0,
            medium_updates=25,
        )
