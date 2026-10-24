"""Shared internal utilities."""

import csv
import json
import os


def init_output_dir(out_dir: str, config: dict, check_existing: bool = True) -> None:
    """
    Create output directory and save inference_settings.json.

    Args:
        out_dir: Path to the output directory
        config: Inference config dict to save
        check_existing: If True, raise ValueError on config mismatch with existing file.
                        If False, silently skip writing if file already exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'inference_settings.json')

    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = json.load(f)
        if check_existing and existing != config:
            raise ValueError(
                f"Current settings do not match previous settings in {config_path}. Aborting."
            )
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def load_processed_ids(path: str, file_type: str = 'csv') -> set:
    """
    Load already-processed custom_id values from a file.

    Args:
        path: Path to the CSV or JSONL file
        file_type: 'csv' or 'jsonl'

    Returns:
        Set of custom_id strings
    """
    processed = set()
    if not os.path.exists(path):
        return processed

    if file_type == 'csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'custom_id' in row:
                    processed.add(row['custom_id'])
    elif file_type == 'jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        processed.add(record['custom_id'])
                    except (json.JSONDecodeError, KeyError):
                        pass

    if processed:
        print(f"Found {len(processed)} already-processed items in {os.path.basename(path)}")
    return processed
