"""
Configuration management utilities for the HOG/LBP/SVM pipeline.
"""

import os
import yaml
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

class ConfigLoader:
    """
    Loads and validates pipeline configuration from YAML files.
    
    Provides type-safe access to configuration parameters and handles
    environment variable substitution and validation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = None
        self._config_hash = None
        
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing all configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If required configuration is missing
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # Perform environment variable substitution
        self._config = self._substitute_env_vars(self._config)
        
        # Validate configuration
        self._validate_config(self._config)
        
        # Calculate config hash for versioning
        self._config_hash = self._calculate_hash(self._config)
        
        return self._config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., "training.svm.C")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get("training.svm.C")
            1.0
            >>> config.get("paths.input_dir") 
            "input"
        """
        if self._config is None:
            self.load()
            
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Top-level section name
            
        Returns:
            Dictionary containing section configuration
        """
        if self._config is None:
            self.load()
            
        if self._config is None:
            return {}
            
        return self._config.get(section, {})
    
    def update(self, key_path: str, value: Any) -> None:
        """
        Update configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
        """
        if self._config is None:
            self.load()
            
        if self._config is None:
            return
            
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if not isinstance(config, dict):
                return
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        if isinstance(config, dict):
            config[keys[-1]] = value
            
            # Recalculate hash
            self._config_hash = self._calculate_hash(self._config)
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Optional path to save config (defaults to original path)
        """
        if self._config is None:
            raise ValueError("No configuration loaded")
            
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get_hash(self) -> str:
        """
        Get hash of current configuration for versioning.
        
        Returns:
            SHA-256 hash of configuration content
        """
        if self._config_hash is None and self._config is not None:
            self._config_hash = self._calculate_hash(self._config)
        return self._config_hash or ""
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Looks for ${VAR_NAME} patterns and replaces with environment variable values.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Simple environment variable substitution
            import re
            def replace_env_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(r'\$\{([^}]+)\}', replace_env_var, config)
        else:
            return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['paths', 'preprocessing', 'features', 'training', 'evaluation']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        # Validate paths section
        paths = config.get('paths', {})
        required_paths = ['input_dir', 'output_dir']
        for path_key in required_paths:
            if path_key not in paths:
                raise ValueError(f"Required path configuration missing: paths.{path_key}")
        
        # Validate preprocessing parameters
        preprocessing = config.get('preprocessing', {})
        if 'target_size' not in preprocessing:
            raise ValueError("Required preprocessing parameter missing: target_size")
        
        target_size = preprocessing['target_size']
        if not isinstance(target_size, list) or len(target_size) != 2:
            raise ValueError("preprocessing.target_size must be a list of [width, height]")
        
        # Validate feature extraction parameters
        features = config.get('features', {})
        if 'hog' not in features or 'lbp' not in features:
            raise ValueError("Both HOG and LBP feature configurations required")
    
    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of configuration for versioning.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Hexadecimal hash string
        """
        # Convert to JSON string for consistent hashing
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()[:16]


@dataclass
class ModelConfig:
    """Type-safe model configuration."""
    kernel: str = "linear"
    C: float = 1.0
    gamma: str = "scale"
    max_iter: int = 1000
    random_state: int = 42
    class_weight: Optional[str] = None


@dataclass 
class HOGConfig:
    """Type-safe HOG feature configuration."""
    orientations: int = 9
    pixels_per_cell: tuple = (8, 8)
    cells_per_block: tuple = (2, 2) 
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = False


@dataclass
class LBPConfig:
    """Type-safe LBP feature configuration."""
    n_points: int = 8
    radius: int = 1
    method: str = "uniform"


class TypedConfig:
    """
    Provides type-safe access to configuration parameters.
    
    Converts dictionary configuration to typed dataclass instances
    for better IDE support and type checking.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize with configuration loader.
        
        Args:
            config_loader: Loaded configuration instance
        """
        self.loader = config_loader
        self._model_config = None
        self._hog_config = None
        self._lbp_config = None
    
    @property
    def model_config(self) -> ModelConfig:
        """Get typed model configuration."""
        if self._model_config is None:
            svm_config = self.loader.get_section('training').get('svm', {})
            self._model_config = ModelConfig(**svm_config)
        return self._model_config
    
    @property
    def hog_config(self) -> HOGConfig:
        """Get typed HOG configuration.""" 
        if self._hog_config is None:
            hog_config = self.loader.get_section('features').get('hog', {})
            # Convert lists to tuples for dataclass
            if 'pixels_per_cell' in hog_config:
                hog_config['pixels_per_cell'] = tuple(hog_config['pixels_per_cell'])
            if 'cells_per_block' in hog_config:
                hog_config['cells_per_block'] = tuple(hog_config['cells_per_block'])
            self._hog_config = HOGConfig(**hog_config)
        return self._hog_config
    
    @property
    def lbp_config(self) -> LBPConfig:
        """Get typed LBP configuration."""
        if self._lbp_config is None:
            lbp_config = self.loader.get_section('features').get('lbp', {})
            self._lbp_config = LBPConfig(**lbp_config)
        return self._lbp_config


def load_config(config_path: str) -> ConfigLoader:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Loaded configuration instance
    """
    loader = ConfigLoader(config_path)
    loader.load()
    return loader


def create_default_config(output_path: str) -> None:
    """
    Create a default configuration file template.
    
    Args:
        output_path: Where to save the default configuration
    """
    # This would create the same config as our YAML file above
    # For now, just copy the existing config.yaml structure
    pass