"""Surgery module for vocabulary and tensor shape adjustments."""

from mergestudio.surgery.vocab import patch_vocab_size_config, reconcile_embedding_shapes

__all__ = ["reconcile_embedding_shapes", "patch_vocab_size_config"]
