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
    
    def get(self, section: str, default: Any = None) -> Dict[str, Any]:
        """Get configuration section (e.g., 'data', 'training', 'windows')."""
        if self._config is None:
            self.load()
        if self._config is None:
            return default
        return self._config.get(section, default)
    
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

    @property
    def optimization(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.get('optimization', {})


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    config = Config(config_path)
    config.load()
    return config
