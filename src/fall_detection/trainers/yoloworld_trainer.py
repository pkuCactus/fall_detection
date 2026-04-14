"""Custom YOLO-World trainer with DDP fix for validation.

Fixes the issue where validation AP is near 0 in DDP mode by ensuring
all ranks have correct text embeddings.
"""

from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_pretrain_routine_end_all_ranks(trainer) -> None:
    """Set up model classes and text encoder on ALL ranks (not just rank 0).

    This fixes the DDP validation issue where AP is near 0 because
    non-zero ranks don't have correct text embeddings.
    """
    # Get class names from dataset
    names = [name.split("/", 1)[0] for name in list(trainer.test_loader.dataset.data["names"].values())]

    # Set classes on both model and EMA for all ranks
    model = unwrap_model(trainer.model)
    ema_model = unwrap_model(trainer.ema.ema) if trainer.ema else None

    # Use cache_clip_model=False to avoid memory issues in DDP
    model.set_classes(names, cache_clip_model=False)
    if ema_model is not None:
        ema_model.set_classes(names, cache_clip_model=False)

    if RANK in {-1, 0}:
        print(f"Set classes for YOLO-World: {names}")


class WorldTrainerDDP(WorldTrainer):
    """YOLO-World trainer with DDP validation fix.

    This trainer ensures all ranks have correct text embeddings,
    fixing the near-0 AP issue in DDP validation.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize trainer and override callback."""
        super().__init__(cfg, overrides, _callbacks)
        # Replace the original callback with our fixed version
        # Use set_callback to properly override (replaces entire callback list)
        self.set_callback("on_pretrain_routine_end", on_pretrain_routine_end_all_ranks)
