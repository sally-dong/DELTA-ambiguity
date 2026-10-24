"""
Repeated trials inference with statistical convergence.

Run the same prompt multiple times until the response distribution converges,
useful for estimating uncertainty when logprobs are not available.
"""

import os
import csv
import math
import pprint
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
from tqdm import tqdm

from .config import Config
from .inference_api import _query_model, _extract_response, clean_response_text


def _has_converged(responses: list, response_type: str, alpha: float) -> bool:
    """
    Check if the distribution estimate has converged within error tolerance alpha.

    Args:
        responses: List of response values
        response_type: 'classification' or 'rating'
        alpha: Convergence threshold (confidence interval width)

    Returns:
        True if converged, False otherwise
    """
    n = len(responses)

    if response_type == 'classification':
        # Binary responses: use Wilson score interval for proportion
        p_hat = sum(int(r) for r in responses) / n
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / n
        margin = z * math.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
        ci_width = 2 * margin
        return ci_width < alpha

    elif response_type == 'rating':
        # Numeric ratings: use standard error of mean
        responses_int = [int(r) for r in responses]
        mean = sum(responses_int) / n
        variance = sum((x - mean)**2 for x in responses_int) / n
        std_error = math.sqrt(variance / n)
        ci_width = 2 * 1.96 * std_error  # 95% CI
        return ci_width < alpha

    return False


def _compute_distribution_stats(responses: list, response_type: str) -> dict:
    """
    Compute distribution statistics from responses.

    Args:
        responses: List of response values
        response_type: 'classification' or 'rating'

    Returns:
        Dictionary with 'mean', 'std', 'num_trials'
    """
    n = len(responses)

    if response_type in ('classification', 'rating'):
        responses_int = [int(r) for r in responses]
        mean = sum(responses_int) / n
        std = math.sqrt(sum((x - mean)**2 for x in responses_int) / n)
    else:
        mean = None
        std = None

    return {'mean': mean, 'std': std, 'num_trials': n}


def _submit_single_trial(inference_settings: dict, response_type: str, inp: dict, trial_num: int) -> tuple[str, str]:
    """
    Submit a single trial and return (custom_id, response_text).

    Returns:
        Tuple of (trial_custom_id, response_text), or (None, None) on failure
    """
    trial_inp = inp.copy()
    trial_inp['custom_id'] = f"{inp['custom_id']}_trial-{trial_num}"

    try:
        response = _query_model(inference_settings, trial_inp)
        response_text, _ = _extract_response(inference_settings, response)

        if response_text is None:
            return None, None

        response_text = clean_response_text(response_type, response_text)

        if response_text is None:
            return None, None

        return trial_inp['custom_id'], response_text

    except Exception as e:
        print(f"Error in trial {trial_num}: {e}")
        return None, None


def _process_single_input_with_trials(out_dir: str, inference_settings: dict, response_type: str, inp: dict, file_lock: Lock) -> None:
    """
    Process a single input with repeated trials until convergence, then write all results.

    Args:
        out_dir: Output directory
        inference_settings: Model configuration with 'alpha', 'min_trials', 'max_trials'
        response_type: 'classification' or 'rating'
        inp: Input dictionary with 'custom_id', 'system_prompt', 'user_prompt'
        file_lock: Threading lock for file writes
    """
    base_id = inp['custom_id']
    alpha = inference_settings['alpha']
    min_trials = inference_settings['min_trials']
    max_trials = inference_settings['max_trials']

    responses = []
    trial_results = []  # Store (trial_id, response_text) tuples
    trial_num = 0

    # Keep submitting until convergence
    while trial_num < min_trials or (trial_num < max_trials and not _has_converged(responses, response_type, alpha)):
        trial_id, response_text = _submit_single_trial(inference_settings, response_type, inp, trial_num)
        if response_text is None:
            trial_num += 1
            continue
        trial_results.append((trial_id, response_text))
        responses.append(response_text)
        trial_num += 1

    # Compute distribution statistics
    stats = _compute_distribution_stats(responses, response_type)

    # Write all results atomically once converged
    with file_lock:
        # Write raw trial results
        with open(os.path.join(out_dir, 'results_raw.csv'), 'a', newline='') as raw_file:
            raw_writer = csv.writer(raw_file)
            for trial_id, response_text in trial_results:
                raw_writer.writerow([trial_id, response_text])
            raw_file.flush()

        # Write converged result with distribution stats
        with open(os.path.join(out_dir, 'results_converged.csv'), 'a', newline='') as conv_file:
            conv_writer = csv.writer(conv_file)
            conv_writer.writerow([base_id, stats['mean'], stats['std'], stats['num_trials']])
            conv_file.flush()


def _load_completed_inputs(out_dir: str) -> set:
    """
    Load the set of completed input IDs from the converged results file.
    Initializes output files if they don't exist.
    """
    from ._utils import load_processed_ids

    converged_path = os.path.join(out_dir, 'results_converged.csv')
    completed = load_processed_ids(converged_path, file_type='csv')

    if not os.path.exists(converged_path):
        with open(converged_path, 'w') as f:
            csv.writer(f).writerow(['custom_id', 'mean', 'std', 'num_trials'])
        with open(os.path.join(out_dir, 'results_raw.csv'), 'w') as f:
            csv.writer(f).writerow(['custom_id', 'Result'])

    return completed


def run_inference_repeated_trials(out_dir: str, inputs: list[dict], inference_settings: dict, response_type: str):
    """
    Run all queries with repeated trials until response distribution converges.

    Each input is repeated a variable number of times until the cumulative response
    distribution converges (based on confidence interval width).

    Answers are immediately written to file, so the function can be rerun without
    repeating already-completed queries (idempotent).

    Args:
        out_dir: Directory where all data and results are saved
        inputs: List of dictionaries, each containing:
            - custom_id: a unique identifier for the input
            - system_prompt: the system prompt for the input
            - user_prompt: the user prompt for the input
        inference_settings: Model configuration with required keys:
            - repeated_trials: Must be True
            - alpha: Convergence threshold (confidence interval width)
            - min_trials: Minimum number of trials per input
            - max_trials: Maximum number of trials per input
        response_type: One of 'rating' (1 numerical digit) or 'classification' (yes/no)
    """
    assert response_type in ['rating', 'classification'], f"Invalid response type: {response_type}"
    assert inference_settings.get('repeated_trials'), "repeated_trials setting must be True"
    assert set(['alpha', 'min_trials', 'max_trials']).issubset(inference_settings), "repeated trials settings not set correctly"
    pprint.pprint(inference_settings)

    from ._utils import init_output_dir
    config = {**inference_settings, 'response_type': response_type, 'source': 'api'}
    init_output_dir(out_dir, config, check_existing=True)

    assert 'trial' not in inputs[0]['custom_id'], "`custom_id` cannot contain the key `trial`, because we are using variable trials"

    from .batch_utils import filter_remaining_prompts

    # Load completed inputs and filter out already processed ones
    completed_ids = _load_completed_inputs(out_dir)
    remaining_inputs = filter_remaining_prompts(inputs, completed_ids)

    print(f"Total inputs: {len(inputs)}, Already completed: {len(completed_ids)}, Remaining: {len(remaining_inputs)}")
    if not remaining_inputs:
        print("All inputs already processed")
        return

    # Prepare query functions
    process_single_input_fn = partial(_process_single_input_with_trials, out_dir, inference_settings, response_type)
    file_lock = Lock()

    # Process remaining inputs in parallel
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_input_fn, inp, file_lock): inp
            for inp in remaining_inputs
        }

        for future in tqdm(as_completed(futures), total=len(remaining_inputs), desc="Processing inputs"):
            inp = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {inp['custom_id']}: {e}")
