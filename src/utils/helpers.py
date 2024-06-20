import os
from pathlib import Path
import pandas as pd
from .config import load_config


def read_data(key: str) -> pd.DataFrame:
    config = load_config()
    file_path = config['data_paths'].get(key)
    if not file_path:
        raise ValueError(f"Path for key '{key}' not found in configuration file.")
    
    project_root = Path('/app')
    full_path = project_root / file_path
    return pd.read_csv(full_path)
