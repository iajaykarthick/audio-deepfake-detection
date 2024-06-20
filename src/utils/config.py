import os
import yaml
from pathlib import Path


def find_project_root(current_path: Path) -> Path:
    for parent in current_path.parents:
        if (parent / '.git').exists():
            return parent
    return current_path


def load_config() -> dict:
    project_root = find_project_root(Path(__file__).resolve().parent)
    config_path = project_root / 'config.yaml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    for key, path in config.get('data_paths', {}).items():
        full_path = project_root / path
        os.makedirs(full_path, exist_ok=True)  
        config['data_paths'][key] = str(full_path)

    return config    
