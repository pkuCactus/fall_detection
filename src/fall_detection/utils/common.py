"""通用工具函数."""

import os
from typing import Dict, Any

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置到YAML文件."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def normalize_device(device: str | None) -> str | None:
    """Normalize device string to proper format.

    Args:
        device: Input device string (e.g., '0', 'cuda:0', 'cpu', None)

    Returns:
        Normalized device string or None
    """
    if device is None:
        return None
    device = str(device).strip()
    if device == '':
        return None
    # Pure number like '0', '1' -> 'cuda:0', 'cuda:1'
    if device.isdigit():
        return f'cuda:{device}'
    # 'cuda' without index -> 'cuda:0'
    if device.lower() == 'cuda':
        return 'cuda:0'
    return device
