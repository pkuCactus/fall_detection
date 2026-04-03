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
