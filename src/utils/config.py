"""
Simplified configuration management for the nurdle detection pipeline.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Simple configuration loader for the nurdle detection pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize config loader with path to YAML file."""
        self.config_path = Path(config_path)
        self._config = None
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        return self._config
    
    def get(self, section: str) -> Dict[str, Any]:
        """Get configuration section (e.g., 'data', 'training', 'windows')."""
        if self._config is None:
            self.load()
        if self._config is None:
            return {}
        return self._config.get(section, {})

    def apply_overrides(self, overrides: Dict[str, Any], allowed_paths=None) -> None:
        """
        Merge a dict of overrides into the loaded config.

        allowed_paths: optional set of dotted paths (e.g., {"data.input_dir", "features.hog_cell_size"}).
        If provided, only these paths (or their children) will be applied.
        """
        if self._config is None:
            self.load()

        def _allowed(path: str) -> bool:
            if not allowed_paths:
                return True
            return any(path == a or a.startswith(path + ".") or path.startswith(a + ".") for a in allowed_paths)

        def _merge(target: Dict[str, Any], patch: Dict[str, Any], prefix: str = ""):
            for key, value in patch.items():
                path = f"{prefix}.{key}" if prefix else key
                if not _allowed(path):
                    continue
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    _merge(target[key], value, path)
                else:
                    target[key] = value

        _merge(self._config, overrides)
    
    @property 
    def data(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.get('data')
    
    @property
    def windows(self) -> Dict[str, Any]:
        """Get window processing configuration."""
        return self.get('windows')
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature extraction configuration."""
        return self.get('features')
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training')
    
    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation')


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config = Config(config_path)
    config.load()
    return config
