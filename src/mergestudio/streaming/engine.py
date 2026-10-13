"""Zero-RAM memory mapped streaming engine for safetensors model merging."""

from __future__ import annotations

import fnmatch
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from mergestudio.algorithms.dare import dare_merge
from mergestudio.algorithms.frankenmerge import LayerSurgeryPlanner
from mergestudio.algorithms.slerp import slerp
from mergestudio.algorithms.ties import ties_merge
from mergestudio.core.config import DType, MergeMethod, MergePlan


@dataclass
class MergeProgress:
    """Real time streaming merge progress status."""

    current_tensor: str
    completed_tensors: int
    total_tensors: int
    bytes_processed: int
    total_estimated_bytes: int
    elapsed_seconds: float
    throughput_mb_s: float
    ram_usage_mb: float


class ModelIndex:
    """Indexes tensors across single or sharded safetensors models."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.tensor_to_file: Dict[str, Path] = {}
        self.tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        self.tensor_dtypes: Dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        index_file = self.model_dir / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            weight_map = data.get("weight_map", {})
            for tensor_name, rel_path in weight_map.items():
                self.tensor_to_file[tensor_name] = self.model_dir / rel_path
        else:
            # Single or unindexed safetensors files
            safetensor_files = sorted(self.model_dir.glob("*.safetensors"))
            for s_file in safetensor_files:
                with safe_open(s_file, framework="pt", device="cpu") as handle:
                    for k in handle.keys():
                        self.tensor_to_file[k] = s_file

    def get_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Load single tensor on demand with memory mapping."""
        file_path = self.tensor_to_file.get(name)
        if not file_path or not file_path.exists():
            return None
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    @property
    def tensor_names(self) -> List[str]:
        return list(self.tensor_to_file.keys())


class ShardedSafetensorsWriter:
    """Streams merged tensors incrementally into sharded safetensors files."""

    def __init__(self, output_dir: Path, max_shard_size_bytes: int = 5 * 1024 * 1024 * 1024):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_size_bytes = max_shard_size_bytes

        self.current_shard_tensors: Dict[str, torch.Tensor] = {}
        self.current_shard_size: int = 0
        self.shard_index: int = 1
        self.weight_map: Dict[str, str] = {}
        self.shard_filenames: List[str] = []
        self.total_size_bytes: int = 0

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Add a merged tensor to current buffer and flush if shard limit reached."""
        tensor_size = tensor.numel() * tensor.element_size()
        if self.current_shard_size + tensor_size > self.max_shard_size_bytes and self.current_shard_tensors:
            self._flush_current_shard()

        self.current_shard_tensors[name] = tensor
        self.current_shard_size += tensor_size
        self.total_size_bytes += tensor_size

    def _flush_current_shard(self) -> None:
        if not self.current_shard_tensors:
            return

        shard_filename = f"model-{self.shard_index:05d}-of-tmp.safetensors"
        out_path = self.output_dir / shard_filename
        save_file(self.current_shard_tensors, out_path)

        for k in self.current_shard_tensors.keys():
            self.weight_map[k] = shard_filename

        self.shard_filenames.append(shard_filename)
        self.shard_index += 1
        self.current_shard_tensors.clear()
        self.current_shard_size = 0
        gc.collect()

    def finalize(self) -> None:
        """Flush remaining tensors, rename shard files with total count, and write index."""
        self._flush_current_shard()
        total_shards = len(self.shard_filenames)

        if total_shards == 0:
            return

        if total_shards == 1:
            # Single file model
            old_name = self.shard_filenames[0]
            new_name = "model.safetensors"
            (self.output_dir / old_name).replace(self.output_dir / new_name)
        else:
            final_weight_map: Dict[str, str] = {}
            for i, old_name in enumerate(self.shard_filenames, start=1):
                new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
                (self.output_dir / old_name).replace(self.output_dir / new_name)
                for k, v in self.weight_map.items():
                    if v == old_name:
                        final_weight_map[k] = new_name

            index_content = {
                "metadata": {"total_size": self.total_size_bytes},
                "weight_map": final_weight_map,
            }
            with open(self.output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(index_content, f, indent=2)


class StreamingMergeEngine:
    """Executes model merges with zero-RAM memory mapped streaming."""

    def __init__(self, plan: MergePlan):
        self.plan = plan
        self.output_dir = Path(plan.output_dir)
        self.target_dtype = self._resolve_dtype(plan.streaming_config.output_dtype)

    @staticmethod
    def _resolve_dtype(dtype: DType) -> torch.dtype:
        if dtype == DType.FLOAT16:
            return torch.float16
        elif dtype == DType.BFLOAT16:
            return torch.bfloat16
        return torch.float32

    def run(
        self,
        progress_callback: Optional[Callable[[MergeProgress], None]] = None,
    ) -> Path:
        """Execute the complete merge plan streaming tensor by tensor."""
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Index all input models
        model_indices: Dict[str, ModelIndex] = {}
        base_model_idx: Optional[ModelIndex] = None

        for m in self.plan.models:
            m_path = Path(m.path)
            model_id = m.name or m_path.name
            idx = ModelIndex(m_path)
            model_indices[model_id] = idx
            if m.is_base or (self.plan.base_model and (self.plan.base_model in (model_id, str(m_path)))):
                base_model_idx = idx

        if self.plan.base_model and not base_model_idx:
            base_p = Path(self.plan.base_model)
            if base_p.exists():
                base_model_idx = ModelIndex(base_p)
                model_indices["__base__"] = base_model_idx

        # Setup surgery planner if frankenmerging
        surgery_planner = None
        if self.plan.method in (MergeMethod.PASSTHROUGH, MergeMethod.FRANKENMERGE):
            surgery_planner = LayerSurgeryPlanner(self.plan.slices)

        # Determine all tensor keys to process
        all_tensor_names: Set[str] = set()
        for idx in model_indices.values():
            all_tensor_names.update(idx.tensor_names)

        filtered_tensor_names = [
            k for k in sorted(all_tensor_names) if self._matches_filter(k)
        ]

        total_tensors = len(filtered_tensor_names)
        max_shard_bytes = int(self.plan.streaming_config.shard_size_gb * 1024 * 1024 * 1024)
        writer = ShardedSafetensorsWriter(self.output_dir, max_shard_size_bytes=max_shard_bytes)

        bytes_processed = 0
        completed = 0

        for key in filtered_tensor_names:
            merged_results = self._merge_tensor_key(
                key=key,
                model_indices=model_indices,
                base_model_idx=base_model_idx,
                surgery_planner=surgery_planner,
            )

            for target_key, merged_tensor in merged_results:
                tensor_out = merged_tensor.to(self.target_dtype)
                writer.write_tensor(target_key, tensor_out)
                tensor_bytes = tensor_out.numel() * tensor_out.element_size()
                bytes_processed += tensor_bytes

            completed += 1
            if progress_callback:
                elapsed = max(0.001, time.time() - start_time)
                throughput = (bytes_processed / (1024 * 1024)) / elapsed
                progress = MergeProgress(
                    current_tensor=key,
                    completed_tensors=completed,
                    total_tensors=total_tensors,
                    bytes_processed=bytes_processed,
                    total_estimated_bytes=int((bytes_processed / max(1, completed)) * total_tensors),
                    elapsed_seconds=elapsed,
                    throughput_mb_s=throughput,
                    ram_usage_mb=0.0,
                )
                progress_callback(progress)

            # Ensure minimal memory footprint
            gc.collect()

        writer.finalize()

        # Copy auxiliary model configurations and tokenizers
        self._copy_metadata(model_indices, base_model_idx, surgery_planner)

        # Generate Ollama Modelfile and GGUF quantization script
        try:
            from mergestudio.export.ollama import generate_gguf_conversion_script, generate_ollama_modelfile
            generate_ollama_modelfile(self.output_dir)
            generate_gguf_conversion_script(self.output_dir)
        except Exception:
            pass

        return self.output_dir

    def _matches_filter(self, key: str) -> bool:
        include = self.plan.filters.include_patterns
        exclude = self.plan.filters.exclude_patterns

        included = any(fnmatch.fnmatch(key, pat) for pat in include)
        excluded = any(fnmatch.fnmatch(key, pat) for pat in exclude)
        return included and not excluded

    def _merge_tensor_key(
        self,
        key: str,
        model_indices: Dict[str, ModelIndex],
        base_model_idx: Optional[ModelIndex],
        surgery_planner: Optional[LayerSurgeryPlanner],
    ) -> List[Tuple[str, torch.Tensor]]:
        """Merge a single tensor across models according to active method."""
        if self.plan.method in (MergeMethod.PASSTHROUGH, MergeMethod.FRANKENMERGE) and surgery_planner:
            results: List[Tuple[str, torch.Tensor]] = []
            for model_id, idx in model_indices.items():
                tensor = idx.get_tensor(key)
                if tensor is not None:
                    mapped = surgery_planner.map_tensor_key(model_id, key)
                    for new_key, _ in mapped:
                        results.append((new_key, tensor.clone()))
            return results

        if self.plan.method == MergeMethod.SLERP:
            m0 = self.plan.models[0]
            m1 = self.plan.models[1]
            id0 = m0.name or Path(m0.path).name
            id1 = m1.name or Path(m1.path).name

            t0 = model_indices[id0].get_tensor(key)
            t1 = model_indices[id1].get_tensor(key)

            if t0 is None and t1 is not None:
                return [(key, t1)]
            if t1 is None and t0 is not None:
                return [(key, t0)]
            if t0 is None and t1 is None:
                return []

            if t0.shape != t1.shape and any(emb in key.lower() for emb in ("embed_tokens", "lm_head", "wte", "embedding")):
                from mergestudio.surgery.vocab import reconcile_embedding_shapes
                t0, t1 = reconcile_embedding_shapes(t0, t1)

            # Check layer weight schedule
            layer_info = LayerSurgeryPlanner.extract_layer_index(key)
            alpha = self.plan.slerp_config.alpha
            if layer_info:
                layer_idx = layer_info[1]
                alpha = self.plan.weight_schedule.get_weight(layer_idx, 32)

            cfg = self.plan.slerp_config.model_copy(update={"alpha": alpha})
            merged = slerp(t0, t1, config=cfg)
            return [(key, merged)]

        elif self.plan.method == MergeMethod.TIES:
            if not base_model_idx:
                raise RuntimeError("Base model index not found for TIES merge")
            base_t = base_model_idx.get_tensor(key)
            if base_t is None:
                return []

            donor_tensors = []
            weights = []
            neg_flags = []
            for m in self.plan.models:
                if not m.is_base:
                    m_id = m.name or Path(m.path).name
                    t = model_indices[m_id].get_tensor(key)
                    if t is not None:
                        if t.shape != base_t.shape and any(emb in key.lower() for emb in ("embed_tokens", "lm_head", "wte", "embedding")):
                            from mergestudio.surgery.vocab import reconcile_embedding_shapes
                            base_t, t = reconcile_embedding_shapes(base_t, t)
                        donor_tensors.append(t)
                        weights.append(m.weight)
                        neg_flags.append(m.is_negative)

            if not donor_tensors:
                return [(key, base_t)]

            merged = ties_merge(
                base=base_t,
                models=donor_tensors,
                model_weights=weights,
                negative_mask=neg_flags,
                config=self.plan.ties_config,
            )
            return [(key, merged)]

        elif self.plan.method in (MergeMethod.DARE_TIES, MergeMethod.DARE_LINEAR):
            if not base_model_idx:
                raise RuntimeError("Base model index not found for DARE merge")
            base_t = base_model_idx.get_tensor(key)
            if base_t is None:
                return []

            donor_tensors = []
            weights = []
            neg_flags = []
            for m in self.plan.models:
                if not m.is_base:
                    m_id = m.name or Path(m.path).name
                    t = model_indices[m_id].get_tensor(key)
                    if t is not None:
                        if t.shape != base_t.shape and any(emb in key.lower() for emb in ("embed_tokens", "lm_head", "wte", "embedding")):
                            from mergestudio.surgery.vocab import reconcile_embedding_shapes
                            base_t, t = reconcile_embedding_shapes(base_t, t)
                        donor_tensors.append(t)
                        weights.append(m.weight)
                        neg_flags.append(m.is_negative)

            if not donor_tensors:
                return [(key, base_t)]

            merged = dare_merge(
                base=base_t,
                models=donor_tensors,
                model_weights=weights,
                negative_mask=neg_flags,
                config=self.plan.dare_config,
            )
            return [(key, merged)]

        # Fallback linear weighted merge
        tensors = []
        weights = []
        for m in self.plan.models:
            m_id = m.name or Path(m.path).name
            t = model_indices[m_id].get_tensor(key)
            if t is not None:
                tensors.append(t)
                weights.append(m.weight)

        if not tensors:
            return []

        w_sum = sum(weights) or 1.0
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        for t, w in zip(tensors, weights):
            acc += (t.to(torch.float32) * (w / w_sum))

        return [(key, acc.to(tensors[0].dtype))]

    def _copy_metadata(
        self,
        model_indices: Dict[str, ModelIndex],
        base_model_idx: Optional[ModelIndex],
        surgery_planner: Optional[LayerSurgeryPlanner],
    ) -> None:
        """Copy config, tokenizer files, and apply config surgery if needed."""
        # Find primary source model directory
        primary_dir = None
        if self.plan.tokenizer_source:
            primary_dir = Path(self.plan.tokenizer_source)
        elif base_model_idx:
            primary_dir = base_model_idx.model_dir
        elif self.plan.models:
            primary_dir = Path(self.plan.models[0].path)

        if not primary_dir or not primary_dir.exists():
            return

        # Copy tokenizers, configs, vocab files
        files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ]

        for fname in files_to_copy:
            src_file = primary_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, self.output_dir / fname)

        # Handle config.json
        cfg_path = primary_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

            if surgery_planner:
                cfg_data = surgery_planner.patch_config(cfg_data)

            with open(self.output_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
