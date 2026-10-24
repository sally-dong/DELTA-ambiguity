"""
Modular batching utilities for large-scale LLM inference experiments.

This module provides reusable functions to:
1. Split large input lists into manageable batches
2. Calculate batch ranges based on batch index
3. Create batch-specific output directories
4. Combine results from multiple batch directories
5. Track completed work across batch directories (robust to prompt reordering)

The "remaining work" batching approach:
- Scans all existing batch_*/probs.csv to find completed custom_ids
- Filters prompts to get only remaining (uncompleted) work
- Batches the remaining work for parallel processing via SLURM array jobs
- Batch index is determined by SLURM_ARRAY_TASK_ID (not user-specified)
- Each batch writes to its own directory to avoid concurrency issues
"""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Any


def get_batch_range(total_items: int, batch_size: int, batch_index: int) -> tuple[int, int]:
    """
    Calculate the start and end indices for a specific batch.

    Args:
        total_items: Total number of items
        batch_size: Number of items per batch
        batch_index: 0-based index of the batch to retrieve

    Returns:
        Tuple of (start_index, end_index) for slicing

    Raises:
        ValueError: If batch_index is out of range
    """
    num_batches = math.ceil(total_items / batch_size)
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError(
            f"Batch index {batch_index} out of range. "
            f"Valid range: 0 to {num_batches - 1} (total batches: {num_batches})"
        )

    start_idx = batch_index * batch_size
    end_idx = min((batch_index + 1) * batch_size, total_items)
    return start_idx, end_idx


def get_batch(items: list[Any], batch_size: int, batch_index: int) -> list[Any]:
    """
    Get a specific batch from a list of items.

    Args:
        items: Full list of items
        batch_size: Number of items per batch
        batch_index: 0-based index of the batch to retrieve

    Returns:
        Subset of items for the specified batch
    """
    start_idx, end_idx = get_batch_range(len(items), batch_size, batch_index)
    return items[start_idx:end_idx]


def get_batch_output_dir(base_output_dir: str, batch_index: int) -> str:
    """
    Generate a batch-specific output directory path.

    Args:
        base_output_dir: Base directory for all outputs
        batch_index: 0-based index of the batch

    Returns:
        Path to batch-specific output directory

    Example:
        >>> get_batch_output_dir("results/my_exp", 0)
        'results/my_exp/batch_000'
    """
    return f"{base_output_dir}/batch_{batch_index:03d}"


def add_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add standard batching arguments to an argument parser.

    The new batching approach:
    - Scans existing batch_*/probs.csv to find completed work
    - Filters to remaining prompts
    - Uses SLURM_ARRAY_TASK_ID to determine which batch to process
    - No manual --batch-index option (prevents user error)

    Args:
        parser: ArgumentParser instance to add arguments to

    Returns:
        Modified parser with batch arguments added

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser = add_batch_args(parser)
        >>> args = parser.parse_args(['--batch-size', '50000'])
    """
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50000,
        help='Number of prompts per batch (default: 50000)'
    )
    parser.add_argument(
        '--total-batches',
        action='store_true',
        help='Print total number of batches needed for remaining work and exit'
    )
    parser.add_argument(
        '--run-remaining',
        action='store_true',
        help='Run all remaining prompts at once (ignoring batch boundaries) and save to next available batch folder'
    )
    return parser


def print_batch_info(total_items: int, batch_size: int, batch_index: int = None) -> None:
    """
    Print informative batch statistics.

    Args:
        total_items: Total number of items to process
        batch_size: Number of items per batch
        batch_index: Optional batch index being processed (None = all batches)
    """
    num_batches = math.ceil(total_items / batch_size)

    print(f"\n{'='*60}")
    print(f"BATCH CONFIGURATION")
    print(f"{'='*60}")
    print(f"Total items:      {total_items:,}")
    print(f"Batch size:       {batch_size:,}")
    print(f"Total batches:    {num_batches}")

    if batch_index is not None:
        start_idx, end_idx = get_batch_range(total_items, batch_size, batch_index)
        batch_items = end_idx - start_idx
        print(f"Current batch:    {batch_index} (0-indexed)")
        print(f"Items in batch:   {batch_items:,} (indices {start_idx:,} to {end_idx-1:,})")
    else:
        print(f"Mode:             Processing all batches together (no batching)")
    print(f"{'='*60}\n")


# --- Remaining work utilities ---

def load_completed_ids_from_batches(base_dir: str, file_type: str = "csv") -> set[str]:
    """
    Scan all batch directories and collect completed custom_ids.

    Args:
        base_dir: Base directory containing batch_* subdirectories
        file_type: Either "csv" (for probs.csv) or "jsonl" (for responses.jsonl)

    Returns:
        Set of custom_ids that have already been processed
    """
    completed_ids: set[str] = set()

    if file_type == "csv":
        pattern = os.path.join(base_dir, "batch_*", "probs.csv")
    else:
        pattern = os.path.join(base_dir, "batch_*", "responses.jsonl")

    batch_files = glob.glob(pattern)

    for batch_file in batch_files:
        if file_type == "csv":
            with open(batch_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'custom_id' in row:
                        completed_ids.add(row['custom_id'])
        else:
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            if 'custom_id' in record:
                                completed_ids.add(record['custom_id'])
                        except json.JSONDecodeError:
                            pass

    return completed_ids


def filter_remaining_prompts(
    prompts: list[dict[str, Any]],
    completed_ids: set[str]
) -> list[dict[str, Any]]:
    """
    Filter out prompts that have already been completed.

    Args:
        prompts: Full list of prompts (each with 'custom_id' key)
        completed_ids: Set of custom_ids that are already done

    Returns:
        List of prompts that still need to be processed
    """
    return [p for p in prompts if p['custom_id'] not in completed_ids]


def get_slurm_batch_index() -> int:
    """
    Get the batch index from SLURM_ARRAY_TASK_ID environment variable.

    Returns:
        Batch index (0 if not running in a SLURM array job)
    """
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        return int(task_id)
    return 0


def get_next_batch_index(base_dir: str) -> int:
    """
    Find the next available batch index by scanning existing batch_* directories.

    Args:
        base_dir: Base directory containing batch_* subdirectories

    Returns:
        Next available batch index (0 if no batches exist)
    """
    batch_pattern = os.path.join(base_dir, "batch_*")
    existing_batches = glob.glob(batch_pattern)

    if not existing_batches:
        return 0

    # Extract batch numbers from directory names
    batch_numbers = []
    for batch_dir in existing_batches:
        dirname = os.path.basename(batch_dir)
        if dirname.startswith("batch_"):
            try:
                batch_num = int(dirname.split("_")[1])
                batch_numbers.append(batch_num)
            except (IndexError, ValueError):
                pass

    if not batch_numbers:
        return 0

    return max(batch_numbers) + 1


def get_remaining_batch(
    remaining_prompts: list[dict[str, Any]],
    batch_size: int,
    batch_index: int
) -> list[dict[str, Any]]:
    """
    Get a specific batch from the remaining prompts.

    Args:
        remaining_prompts: List of prompts still needing processing
        batch_size: Number of items per batch
        batch_index: Which batch to retrieve (from SLURM_ARRAY_TASK_ID)

    Returns:
        Subset of remaining prompts for this batch, or empty list if batch_index
        is beyond the available work
    """
    start_idx = batch_index * batch_size
    end_idx = min((batch_index + 1) * batch_size, len(remaining_prompts))

    if start_idx >= len(remaining_prompts):
        return []

    return remaining_prompts[start_idx:end_idx]


def print_remaining_batch_info(
    total_prompts: int,
    completed_count: int,
    remaining_count: int,
    batch_size: int,
    batch_index: int,
    batch_items: int
) -> None:
    """
    Print informative statistics about remaining work batching.

    Args:
        total_prompts: Total number of prompts in the experiment
        completed_count: Number of already-completed prompts
        remaining_count: Number of prompts still to process
        batch_size: Number of items per batch
        batch_index: Current batch index (from SLURM_ARRAY_TASK_ID)
        batch_items: Number of items in this specific batch
    """
    num_batches = math.ceil(remaining_count / batch_size) if remaining_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"REMAINING WORK BATCH CONFIGURATION")
    print(f"{'='*60}")
    print(f"Total prompts:    {total_prompts:,}")
    print(f"Already done:     {completed_count:,}")
    print(f"Remaining:        {remaining_count:,}")
    print(f"Batch size:       {batch_size:,}")
    print(f"Batches needed:   {num_batches}")
    print(f"Current batch:    {batch_index} (from SLURM_ARRAY_TASK_ID)")
    print(f"Items this batch: {batch_items:,}")
    print(f"{'='*60}\n")


# --- Batch combination utilities ---

def combine_jsonl_batches(base_dir: str, verbose: bool = True) -> None:
    """
    Combine all responses.jsonl files from batch directories into a single file.

    Args:
        base_dir: Base directory containing batch_* subdirectories
        output_file: Path to write combined results
        verbose: Print progress information
    """
    batch_pattern = os.path.join(base_dir, "batch_*", "responses.jsonl")
    output_file = os.path.join(base_dir, "probs.csv")
    
    batch_files = sorted(glob.glob(batch_pattern))

    if not batch_files:
        print(f"Error: No batch files found matching {batch_pattern}")
        return

    if verbose:
        print(f"Found {len(batch_files)} batch files to combine:")
        for f in batch_files:
            print(f"  - {f}")
        print()

    total_records = 0
    seen_ids = set()
    duplicates = 0

    with open(output_file, 'w', encoding='utf-8') as outf:
        for batch_file in batch_files:
            if verbose:
                print(f"Processing {batch_file}...")

            batch_records = 0
            with open(batch_file, 'r', encoding='utf-8') as inf:
                for line in inf:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        custom_id = record.get('custom_id')

                        if custom_id in seen_ids:
                            duplicates += 1
                            if verbose:
                                print(f"  Warning: Duplicate ID found: {custom_id}")
                            continue

                        seen_ids.add(custom_id)
                        outf.write(line + '\n')
                        batch_records += 1
                        total_records += 1

                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipping malformed line in {batch_file}: {e}")

            if verbose:
                print(f"  Added {batch_records} records")

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMBINATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total records written: {total_records:,}")
        print(f"Duplicates skipped:    {duplicates:,}")
        print(f"Output file:           {output_file}")
        print(f"{'='*60}\n")


def combine_csv_batches(base_dir: str, verbose: bool = True) -> None:
    """
    Combine all probs.csv files from batch directories into a single CSV.

    Args:
        base_dir: Base directory containing batch_* subdirectories
        output_file: Path to write combined results
        verbose: Print progress information
    """
    batch_pattern = os.path.join(base_dir, "batch_*", "probs.csv")
    output_file = os.path.join(base_dir, "probs.csv")
    batch_files = sorted(glob.glob(batch_pattern))

    if not batch_files:
        print(f"Error: No batch files found matching {batch_pattern}")
        return

    if verbose:
        print(f"Found {len(batch_files)} batch CSV files to combine:")
        for f in batch_files:
            print(f"  - {f}")
        print()

    total_records = 0
    seen_ids = set()
    duplicates = 0

    with open(output_file, 'w', encoding='utf-8', newline='') as outf:
        writer = None

        for batch_file in batch_files:
            if verbose:
                print(f"Processing {batch_file}...")

            batch_records = 0
            with open(batch_file, 'r', encoding='utf-8') as inf:
                reader = csv.DictReader(inf)

                if writer is None:
                    writer = csv.DictWriter(outf, fieldnames=reader.fieldnames)
                    writer.writeheader()

                for row in reader:
                    custom_id = row.get('custom_id')

                    if custom_id in seen_ids:
                        duplicates += 1
                        #if verbose:
                        #    print(f"  Warning: Duplicate ID found: {custom_id}")
                        continue

                    seen_ids.add(custom_id)
                    writer.writerow(row)
                    batch_records += 1
                    total_records += 1

            if verbose:
                print(f"  Added {batch_records} records")

    if verbose:
        print(f"\n{'='*60}")
        print(f"COMBINATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total records written: {total_records:,}")
        print(f"Duplicates skipped:    {duplicates:,}")
        print(f"Output file:           {output_file}")
        print(f"{'='*60}\n")


def _combine_batches_cli():
    """CLI entry point for combining batch results."""
    parser = argparse.ArgumentParser(
        description='Combine batch results into a single file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine JSONL files
  python -m delta.inference.batch_utils results/my_exp results/combined.jsonl

  # Combine CSV files
  python -m delta.inference.batch_utils results/my_exp results/combined.csv

  # Quiet mode
  python -m delta.inference.batch_utils results/my_exp results/combined.jsonl --quiet
        """
    )
    parser.add_argument(
        'base_dir',
        help='Base directory containing batch_* subdirectories'
    )
    parser.add_argument(
        '--output_type',
        default='csv',
        help='jsonl or csv'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: Directory not found: {args.base_dir}")
        return

    if args.output_type == 'jsonl':
        combine_jsonl_batches(args.base_dir, verbose=not args.quiet)
    elif args.output_type == 'csv':
        combine_csv_batches(args.base_dir, verbose=not args.quiet)
    else:
        print(f"Error: Unsupported file extension '{args.output_type}'. Use .jsonl or .csv")

if __name__ == "__main__":
    _combine_batches_cli()
