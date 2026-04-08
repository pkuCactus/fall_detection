"""Common training utilities shared across training scripts."""

import argparse
import ast
import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import yaml


def parse_args(description: str = "Training script") -> argparse.Namespace:
    """Parse command line arguments.
    
    Args:
        description: Script description for help text
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for DDP")
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config values, e.g., 'epochs=100,batch=8'",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and parse configuration with override support.
    
    Args:
        args: Parsed arguments containing config path and optional overrides
        
    Returns:
        Configuration dictionary
    """
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Process command line overrides
    if args.override:
        for override in args.override.split(","):
            key, value = override.split("=")
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            d[keys[-1]] = value

    return cfg


def setup_ddp(cfg: Dict[str, Any]) -> Tuple[bool, torch.device, int, int, int]:
    """Setup Distributed Data Parallel.
    
    Args:
        cfg: Configuration dictionary containing optional 'ddp' section
        
    Returns:
        Tuple of (ddp_enabled, device, world_size, rank, local_rank)
    """
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if ddp:
        ddp_cfg = cfg.get("ddp", {})
        backend = ddp_cfg.get("backend", "nccl")
        
        # Support custom port from config
        port = ddp_cfg.get("port", None)
        if port:
            os.environ["MASTER_PORT"] = str(port)
            if int(os.environ.get("RANK", 0)) == 0:
                print(f"DDP using custom port: {port}")

        dist.init_process_group(backend=backend if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0
    
    if rank == 0:
        print(f"DDP setup: ddp={ddp}, device={device}, world_size={world_size}, rank={rank}, local_rank={local_rank}")

    return ddp, device, world_size, rank, local_rank


def setup_seed(seed: Optional[int], rank: int = 0) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value (None to skip)
        rank: Process rank (only rank 0 prints message)
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == 0:
        print(f"Random seed set to {seed}")


def should_stop_early(
    cfg: Dict[str, Any],
    rank: int,
    device: torch.device,
    ddp: bool,
    patience_counter: int,
) -> bool:
    """Check if training should stop early (DDP synchronized).
    
    This function synchronizes the stop decision across all DDP ranks.
    
    Args:
        cfg: Configuration dictionary with 'early_stopping' section
        rank: Current process rank
        device: Torch device
        ddp: Whether DDP is enabled
        patience_counter: Current patience counter (only checked on rank 0)
        
    Returns:
        True if all ranks should stop, False otherwise
    """
    early_cfg = cfg.get("early_stopping", {})
    
    # Create stop signal tensor (all ranks)
    should_stop = torch.tensor(0.0, device=device)
    
    # Rank 0 decides based on patience_counter
    if rank == 0 and early_cfg.get("enabled", True):
        if patience_counter >= early_cfg.get("patience", 20):
            should_stop = torch.tensor(1.0, device=device)
    
    # Broadcast to all ranks
    if ddp:
        dist.broadcast(should_stop, src=0)
    
    return should_stop.item() > 0


def format_time_remaining(remaining_secs: int) -> str:
    """Format remaining time as human-readable string.
    
    Args:
        remaining_secs: Remaining seconds
        
    Returns:
        Formatted string like "4m30s" or "1h15m"
    """
    remaining_mins = int(remaining_secs / 60)
    if remaining_mins >= 60:
        remaining_hours = remaining_mins // 60
        remaining_mins_remainder = remaining_mins % 60
        return f"{remaining_hours}h{remaining_mins_remainder}m"
    return f"{remaining_mins}m{int(remaining_secs % 60)}s"
