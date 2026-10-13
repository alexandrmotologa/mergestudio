"""Layer surgery and passthrough frankenmerging for transformer models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mergestudio.core.config import LayerSlice


class LayerSurgeryPlanner:
    """Computes mappings and metadata transforms for layer splicing and frankenmerging."""

    # Matches layer index patterns in transformer weight names
    # Examples: model.layers.12.self_attn.q_proj.weight, transformer.h.3.mlp...
    LAYER_REGEXES = [
        re.compile(r"^(.*(?:layers|h|block|blocks)\.)(\d+)(\..*)$"),
    ]

    def __init__(self, slices: List[LayerSlice]):
        self.slices = slices
        self.total_target_layers = 0
        self.slice_mappings: List[Dict[str, Any]] = []
        self._build_schedule()

    def _build_schedule(self) -> None:
        """Calculate target layer indices and offsets for each slice."""
        current_target_idx = 0
        for s in self.slices:
            slice_len = s.end_layer - s.start_layer
            target_start = s.target_layer_start if s.target_layer_start is not None else current_target_idx
            target_end = target_start + slice_len

            self.slice_mappings.append(
                {
                    "source_model": s.source_model,
                    "source_start": s.start_layer,
                    "source_end": s.end_layer,
                    "target_start": target_start,
                    "target_end": target_end,
                    "length": slice_len,
                }
            )
            current_target_idx = max(current_target_idx, target_end)

        self.total_target_layers = current_target_idx

    @classmethod
    def extract_layer_index(cls, key: str) -> Optional[Tuple[str, int, str]]:
        """Extract (prefix, layer_idx, suffix) from a tensor name."""
        for pattern in cls.LAYER_REGEXES:
            match = pattern.match(key)
            if match:
                prefix, idx_str, suffix = match.groups()
                return prefix, int(idx_str), suffix
        return None

    def map_tensor_key(
        self, source_model_id: str, key: str
    ) -> List[Tuple[str, int]]:
        """Map a source tensor key to output tensor key(s) with new layer index.

        Returns list of (new_key, target_layer_idx). Non-layer weights return [(key, -1)].
        """
        extracted = self.extract_layer_index(key)
        if not extracted:
            # Embedding, norm, or lm_head
            return [(key, -1)]

        prefix, source_idx, suffix = extracted
        mapped_keys: List[Tuple[str, int]] = []

        for mapping in self.slice_mappings:
            if mapping["source_model"] == source_model_id:
                s_start = mapping["source_start"]
                s_end = mapping["source_end"]
                if s_start <= source_idx < s_end:
                    offset = source_idx - s_start
                    t_idx = mapping["target_start"] + offset
                    new_key = f"{prefix}{t_idx}{suffix}"
                    mapped_keys.append((new_key, t_idx))

        return mapped_keys

    def patch_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an updated model configuration dictionary with modified layer counts."""
        updated = json.loads(json.dumps(base_config))  # deep copy

        # Keys commonly used across Hugging Face transformer architectures
        layer_keys = [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "n_layers",
            "num_blocks",
        ]

        patched = False
        for k in layer_keys:
            if k in updated:
                updated[k] = self.total_target_layers
                patched = True

        if not patched:
            updated["num_hidden_layers"] = self.total_target_layers

        return updated
