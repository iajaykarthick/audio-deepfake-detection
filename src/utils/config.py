import os
import yaml
from pathlib import Path


def find_project_root(current_path: Path) -> Path:
    for parent in current_path.parents:
        if (parent / '.git').exists():
            return parent
    return current_path

def resolve_paths(config: dict, root: Path) -> dict:
    """
    Recursively resolve paths in the configuration dictionary.
    """
    resolved_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = resolve_paths(value, root)
        else:
            full_path = root / value
            os.makedirs(full_path, exist_ok=True)
            resolved_config[key] = str(full_path)
    return resolved_config

def load_config() -> dict:
    project_root = find_project_root(Path(__file__).resolve().parent)
    config_path = project_root / 'config.yaml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if 'data_paths' in config:
        config['data_paths'] = resolve_paths(config['data_paths'], project_root)
    
    return config