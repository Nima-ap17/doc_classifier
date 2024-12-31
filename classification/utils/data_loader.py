"""This module contains utility functions for loading data."""
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load the configuration file.
    """
    config_path = Path('classification/config.yaml')
    with config_path.open('r') as f:
        return yaml.safe_load(f)


def load_data(
    x_file_path: str, 
    y1_file_path: str, 
    y2_file_path: str, 
    category_mappings: dict, 
    subcategory_mappings: dict
) -> pd.DataFrame:
    """
    Load the data from the provided files and return a DataFrame.
    param x_file_path: Path to the file containing the text data
    param y1_file_path: Path to the file containing the main category labels
    param y2_file_path: Path to the file containing the subcategory labels
    param category_mappings: Dictionary mapping category labels to category names
    param subcategory_mappings: Dictionary mapping subcategory labels to subcategory names
    return: DataFrame containing the loaded data
    """
    with open(x_file_path) as f:
        x_content = f.readlines()

    with open(y1_file_path) as f:
        y1_labels = [line.strip() for line in f.readlines()]

    with open(y2_file_path) as f:
        y2_labels = [line.strip() for line in f.readlines()]

    # Replace labels with category names
    parent_label = [category_mappings[num] for num in y1_labels]
    child_label = [subcategory_mappings[num] for num in y2_labels]

    return pd.DataFrame({
        'text': x_content,
        'parent_label': parent_label,
        'child_label': child_label
    })
