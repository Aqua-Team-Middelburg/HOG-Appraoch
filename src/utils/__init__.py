"""Utility modules for the pipeline.

Keep imports lightweight so core config loading works even if optional
visualization dependencies (e.g., matplotlib) are not installed.
"""

from .config import Config, load_config

__all__ = ['Config', 'load_config']
