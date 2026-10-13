"""Configuration models and validation schemas for MergeStudio."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class MergeMethod(str, Enum):
    """Supported model merging algorithms."""

    SLERP = "slerp"
    TIES = "ties"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    LINEAR = "linear"
    PASSTHROUGH = "passthrough"
    FRANKENMERGE = "frankenmerge"
    TASK_ARITHMETIC = "task_arithmetic"


class DType(str, Enum):
    """Storage precision datatypes for output tensors."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class WeightScheduleType(str, Enum):
    """Interpolation schedules across model layers."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    SINE = "sine"
    SIGMOID = "sigmoid"
    CUSTOM = "custom"


class ModelSource(BaseModel):
    """Input model reference."""

    path: str = Field(description="Local directory path containing safetensors files")
    name: Optional[str] = Field(default=None, description="Human friendly identifier")
    is_base: bool = Field(default=False, description="Flag indicating if this model is the base anchor")
    is_negative: bool = Field(default=False, description="Flag indicating negative vector unlearning")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weighting coefficient")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model path cannot be empty")
        return v.strip()


class LayerSlice(BaseModel):
    """Definition of layer slice for passthrough frankenmerging."""

    source_model: str = Field(description="Identifier or path of source model")
    start_layer: int = Field(ge=0, description="Inclusive start layer index in source")
    end_layer: int = Field(ge=0, description="Exclusive end layer index in source")
    target_layer_start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Target starting layer index. If None, assigned sequentially.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> LayerSlice:
        if self.end_layer <= self.start_layer:
            raise ValueError(
                f"end_layer ({self.end_layer}) must be strictly greater than start_layer ({self.start_layer})"
            )
        return self


class WeightSchedule(BaseModel):
    """Layer wise weight progression curve."""

    schedule_type: WeightScheduleType = WeightScheduleType.CONSTANT
    start_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    end_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    custom_weights: Optional[List[float]] = None

    def get_weight(self, layer_idx: int, total_layers: int) -> float:
        """Calculate interpolation weight for a specific layer index."""
        if total_layers <= 1:
            return self.start_weight

        if self.schedule_type == WeightScheduleType.CUSTOM and self.custom_weights:
            if layer_idx < len(self.custom_weights):
                return float(self.custom_weights[layer_idx])
            return float(self.custom_weights[-1])

        t = layer_idx / float(total_layers - 1)
        w0 = self.start_weight
        w1 = self.end_weight

        if self.schedule_type == WeightScheduleType.CONSTANT:
            return w0
        elif self.schedule_type == WeightScheduleType.LINEAR:
            return w0 + t * (w1 - w0)
        elif self.schedule_type == WeightScheduleType.COSINE:
            factor = (1.0 - math.cos(t * math.pi)) / 2.0
            return w0 + factor * (w1 - w0)
        elif self.schedule_type == WeightScheduleType.SINE:
            factor = math.sin(t * (math.pi / 2.0))
            return w0 + factor * (w1 - w0)
        elif self.schedule_type == WeightScheduleType.SIGMOID:
            k = 10.0
            sig = 1.0 / (1.0 + math.exp(-k * (t - 0.5)))
            return w0 + sig * (w1 - w0)

        return w0


class SlerpConfig(BaseModel):
    """Configuration parameters for Spherical Linear Interpolation."""

    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Interpolation parameter")
    eps: float = Field(default=1e-7, gt=0.0, description="Threshold for collinear detection")
    per_row: bool = Field(default=False, description="Apply SLERP per row rather than globally")
    preserve_norm: bool = Field(default=True, description="Interpolate magnitude along with direction")


class TiesConfig(BaseModel):
    """Configuration parameters for TIES merging."""

    density: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Fraction of top magnitude weights to retain during trimming (0.0 < density <= 1.0)",
    )
    majority_sign_method: str = Field(
        default="total",
        description="Election algorithm for majority sign: 'total' (sum of deltas) or 'frequency' (vote count)",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Global scaling multiplier lambda applied to the aggregated delta",
    )
    normalize_signs: bool = Field(
        default=True,
        description="Exclude discordant vectors whose signs conflict with the elected majority",
    )


class DareConfig(BaseModel):
    """Configuration parameters for DARE merging."""

    drop_rate: float = Field(
        default=0.5,
        ge=0.0,
        lt=1.0,
        description="Bernoulli drop probability p for zeroing parameter deltas",
    )
    rescale: bool = Field(
        default=True,
        description="Scale surviving deltas by 1 / (1 - drop_rate) to maintain expected value",
    )
    base_method: MergeMethod = Field(
        default=MergeMethod.DARE_TIES,
        description="Underlying reduction method (dare_ties or dare_linear)",
    )


class StreamingConfig(BaseModel):
    """Zero-RAM streaming and memory mapped I/O settings."""

    chunk_size_bytes: int = Field(
        default=1024 * 1024 * 1024,
        gt=0,
        description="Internal buffer size for chunked operations (bytes)",
    )
    shard_size_gb: float = Field(
        default=5.0,
        gt=0.1,
        description="Maximum size in gigabytes per safetensors output shard",
    )
    output_dtype: DType = Field(
        default=DType.FLOAT16,
        description="Precision for exported merged tensors",
    )
    buffer_device: str = Field(
        default="cpu",
        description="Device for tensor compute buffer: 'cpu' or 'cuda'",
    )


class FilterSpec(BaseModel):
    """Layer and tensor key filter patterns."""

    include_patterns: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Glob or regex patterns of tensor keys to include",
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Glob or regex patterns of tensor keys to skip",
    )


class MergePlan(BaseModel):
    """Comprehensive plan for executing a model merge operation."""

    name: str = Field(default="merged-model", description="Identifier for this merge operation")
    method: MergeMethod = Field(description="Selected merging algorithm")
    models: List[ModelSource] = Field(
        default_factory=list,
        description="List of donor and base models",
    )
    base_model: Optional[str] = Field(
        default=None,
        description="Explicit identifier or path of the base anchor model (required for TIES/DARE)",
    )
    slerp_config: SlerpConfig = Field(default_factory=SlerpConfig)
    ties_config: TiesConfig = Field(default_factory=TiesConfig)
    dare_config: DareConfig = Field(default_factory=DareConfig)
    streaming_config: StreamingConfig = Field(default_factory=StreamingConfig)
    filters: FilterSpec = Field(default_factory=FilterSpec)
    slices: List[LayerSlice] = Field(
        default_factory=list,
        description="Layer slices for passthrough frankenmerging",
    )
    weight_schedule: WeightSchedule = Field(default_factory=WeightSchedule)
    output_dir: str = Field(
        default="./output/merged_model",
        description="Destination directory for output model files",
    )
    tokenizer_source: Optional[str] = Field(
        default=None,
        description="Path or model name from which to copy tokenizer configurations and vocabulary",
    )

    @model_validator(mode="after")
    def validate_method_requirements(self) -> MergePlan:
        if self.method == MergeMethod.SLERP:
            if len(self.models) != 2:
                raise ValueError("SLERP merging requires exactly two models")
        elif self.method in (
            MergeMethod.TIES,
            MergeMethod.DARE_TIES,
            MergeMethod.DARE_LINEAR,
            MergeMethod.TASK_ARITHMETIC,
        ):
            has_base = any(m.is_base for m in self.models) or (self.base_model is not None)
            if not has_base:
                raise ValueError(
                    f"{self.method.value} merging requires a base model designated either via is_base=True or base_model"
                )
            if len(self.models) < 2:
                raise ValueError(
                    f"{self.method.value} merging requires at least a base model and one fine-tuned donor model"
                )
        elif self.method in (MergeMethod.PASSTHROUGH, MergeMethod.FRANKENMERGE):
            if not self.slices:
                raise ValueError("Passthrough / Frankenmerge requires at least one layer slice definition")
        return self

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Write merge plan to YAML file."""
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> MergePlan:
        """Load merge plan from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def load_json(cls, data: Union[str, Dict[str, Any]]) -> MergePlan:
        """Load merge plan from JSON string or dictionary."""
        if isinstance(data, str):
            return cls.model_validate_json(data)
        return cls.model_validate(data)
