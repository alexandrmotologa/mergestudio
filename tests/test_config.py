"""Tests for MergeStudio configuration models and validation."""

import pytest
from pydantic import ValidationError

from mergestudio.core.config import (
    DareConfig,
    DType,
    LayerSlice,
    MergeMethod,
    MergePlan,
    ModelSource,
    SlerpConfig,
    TiesConfig,
    WeightSchedule,
    WeightScheduleType,
)


def test_model_source_validation():
    m = ModelSource(path="./models/llama-7b", name="llama", is_base=True, weight=1.0)
    assert m.path == "./models/llama-7b"
    assert m.is_base is True

    with pytest.raises(ValidationError):
        ModelSource(path="")


def test_layer_slice_validation():
    s = LayerSlice(source_model="model_a", start_layer=0, end_layer=16)
    assert s.start_layer == 0
    assert s.end_layer == 16

    with pytest.raises(ValidationError):
        LayerSlice(source_model="model_a", start_layer=10, end_layer=10)

    with pytest.raises(ValidationError):
        LayerSlice(source_model="model_a", start_layer=20, end_layer=10)


def test_weight_schedule_curves():
    # Linear schedule
    sched_linear = WeightSchedule(
        schedule_type=WeightScheduleType.LINEAR,
        start_weight=0.0,
        end_weight=1.0,
    )
    assert sched_linear.get_weight(0, 5) == pytest.approx(0.0)
    assert sched_linear.get_weight(2, 5) == pytest.approx(0.5)
    assert sched_linear.get_weight(4, 5) == pytest.approx(1.0)

    # Cosine schedule
    sched_cos = WeightSchedule(
        schedule_type=WeightScheduleType.COSINE,
        start_weight=0.0,
        end_weight=1.0,
    )
    assert sched_cos.get_weight(0, 5) == pytest.approx(0.0)
    assert sched_cos.get_weight(4, 5) == pytest.approx(1.0)
    # Cosine curve is monotonic
    assert 0.0 < sched_cos.get_weight(2, 5) < 1.0

    # Custom schedule
    sched_custom = WeightSchedule(
        schedule_type=WeightScheduleType.CUSTOM,
        custom_weights=[0.1, 0.4, 0.9],
    )
    assert sched_custom.get_weight(0, 3) == pytest.approx(0.1)
    assert sched_custom.get_weight(1, 3) == pytest.approx(0.4)
    assert sched_custom.get_weight(2, 3) == pytest.approx(0.9)


def test_merge_plan_slerp_validation():
    # SLERP requires exactly 2 models
    with pytest.raises(ValidationError):
        MergePlan(
            method=MergeMethod.SLERP,
            models=[ModelSource(path="m1")],
        )

    plan = MergePlan(
        method=MergeMethod.SLERP,
        models=[
            ModelSource(path="m1", is_base=True),
            ModelSource(path="m2"),
        ],
        slerp_config=SlerpConfig(alpha=0.7),
    )
    assert plan.slerp_config.alpha == 0.7


def test_merge_plan_ties_validation():
    # TIES requires a base model
    with pytest.raises(ValidationError):
        MergePlan(
            method=MergeMethod.TIES,
            models=[
                ModelSource(path="m1", is_base=False),
                ModelSource(path="m2", is_base=False),
            ],
        )

    plan = MergePlan(
        method=MergeMethod.TIES,
        models=[
            ModelSource(path="base", is_base=True),
            ModelSource(path="donor1"),
        ],
        ties_config=TiesConfig(density=0.3),
    )
    assert plan.ties_config.density == 0.3


def test_merge_plan_serialization(tmp_path):
    yaml_file = tmp_path / "plan.yaml"
    plan = MergePlan(
        name="test-slerp",
        method=MergeMethod.SLERP,
        models=[
            ModelSource(path="model1"),
            ModelSource(path="model2"),
        ],
    )
    plan.save_yaml(yaml_file)

    loaded = MergePlan.load_yaml(yaml_file)
    assert loaded.name == "test-slerp"
    assert loaded.method == MergeMethod.SLERP
    assert len(loaded.models) == 2
