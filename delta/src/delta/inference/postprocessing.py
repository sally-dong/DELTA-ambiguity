"""
Post-inference processing.

Converts raw inference outputs into standardized probability files:
1. API logprobs (CSV) → probs.csv via softmax
2. Sampled results (CSV) → probs.csv via trial frequency distributions
3. Auto-dispatch via postprocess() using saved inference_settings.json
"""

import json
import os

import numpy as np
import pandas as pd


DEFAULT_ANSWERS = {
    'classification': [0, 1],
    'rating': list(range(10)),
}


# ---- Custom ID parsing ----

def parse_custom_ids(df):
    """
    Expand a 'custom_id' column into separate columns.

    custom_id format: 'columnname1-value1_columnname2-value2_...'
    Each part becomes its own column with the extracted value.

    Args:
        df: DataFrame with a 'custom_id' column

    Returns:
        (df, column_names) — modified DataFrame (custom_id dropped) and list of new column names
    """
    first_id = df['custom_id'].iloc[0]
    column_names = [part.split('-')[0] for part in first_id.split('_')]
    pattern = '_'.join([f'{col}-([^_]+)' for col in column_names])
    df[column_names] = df['custom_id'].str.extract(pattern)
    df.drop(columns=['custom_id'], inplace=True)
    return df, column_names


def parse_raw_results_file(out_dir: str):
    """Parse 'results_raw.csv' to 'results.csv' by expanding custom_id into columns."""
    results = pd.read_csv(os.path.join(out_dir, 'results_raw.csv'))
    results, _ = parse_custom_ids(results)
    results.to_csv(os.path.join(out_dir, 'results.csv'), index=False)


# ---- Shared output formatting ----

def _format_rating_output(probs_long, groupby_cols, answer_col, answers):
    """
    Convert long-format rating probabilities to wide format with expected value.

    Args:
        probs_long: DataFrame with groupby_cols + [answer_col, 'prob']
        groupby_cols: columns identifying each query
        answer_col: name of the column containing answer values
        answers: list of all possible answer values

    Returns:
        DataFrame with groupby_cols + ['expected_value', 'prob_0', ..., 'prob_N']
    """
    pivot = probs_long.pivot_table(
        index=groupby_cols, columns=answer_col, values='prob', fill_value=0
    )
    pivot.columns = [f'prob_{int(c)}' for c in pivot.columns]
    out = pivot.reset_index()
    for a in answers:
        col = f'prob_{a}'
        if col not in out.columns:
            out[col] = 0.0
    out['expected_value'] = sum(a * out[f'prob_{a}'] for a in answers)
    prob_cols = [f'prob_{a}' for a in answers]
    return out[groupby_cols + ['expected_value'] + prob_cols]


# ---- API logprobs → probabilities ----

def _process_logprob_tokens(results: pd.DataFrame, response_type: str = 'classification') -> pd.DataFrame:
    """Filter logprob tokens to valid answers and add an 'answer' column."""
    token_lower = results['token'].astype(str).str.lower()

    if response_type == 'classification':
        mask = token_lower.str.contains('yes|no', na=False)
        logprobs = results[mask].copy()
        logprobs['answer'] = token_lower[mask].str.contains('yes').astype(int)

    elif response_type == 'rating':
        mask = results['token'].astype(str).str.fullmatch(r'\d', na=False)
        logprobs = results[mask].copy()
        logprobs['answer'] = logprobs['token'].astype(int)

    return logprobs


def logprobs_to_probabilities(results_dir: str, result_file_name: str = 'logprobs_raw.csv',
                              response_type: str = 'classification', answers: list = None) -> None:
    """
    Convert API logprobs CSV into per-answer probabilities via softmax.

    Output format (one row per query):
        - classification: custom_id, prob_yes
        - rating: custom_id, expected_value, prob_0, prob_1, ..., prob_N

    Args:
        results_dir: Directory containing the logprobs file
        result_file_name: Name of the input logprobs file
        response_type: 'classification' or 'rating'
        answers: List of possible answers (default from DEFAULT_ANSWERS)
    """
    if answers is None:
        answers = DEFAULT_ANSWERS[response_type]
        print(f"INFO: Using default answers for '{response_type}': {answers}")

    logprobs = pd.read_csv(os.path.join(results_dir, result_file_name))
    grouping_columns = [c for c in logprobs.columns if c not in {'logprob', 'token'}]

    logprobs = _process_logprob_tokens(logprobs, response_type)

    # Softmax within each query group
    logprobs['prob'] = np.exp(
        logprobs['logprob'] - logprobs.groupby(grouping_columns)['logprob'].transform(np.logaddexp.reduce)
    ).round(5)

    # Sum probabilities per answer (multiple tokens can map to same answer, e.g. "Yes" and "yes")
    probs = logprobs.groupby(grouping_columns + ['answer'], as_index=False)['prob'].sum()

    # Ensure all answers are present for every query
    keys = probs[grouping_columns].drop_duplicates()
    complete = keys.merge(pd.DataFrame({'answer': answers}), how='cross')
    probs = complete.merge(probs, how='left').fillna({'prob': 0})

    if response_type == 'classification':
        out = probs[probs['answer'] == 1][grouping_columns + ['prob']].copy()
        out.rename(columns={'prob': 'prob_yes'}, inplace=True)
    else:
        out = _format_rating_output(probs, grouping_columns, 'answer', answers)

    out.to_csv(os.path.join(results_dir, 'probs.csv'), index=False)


# ---- Sampled results → probabilities (no logprobs) ----

def results_to_probabilities(results_dir: str, response_type: str = 'classification',
                             answers: list = None) -> None:
    """
    Convert results_raw.csv (sampled responses) into probs.csv by computing
    frequency distributions over repeated trials.

    This is the no-logprobs analog of logprobs_to_probabilities().

    Output format matches logprobs_to_probabilities():
        - classification: custom_id, prob_yes
        - rating: custom_id, expected_value, prob_0, prob_1, ..., prob_N

    Args:
        results_dir: Directory containing results_raw.csv
        response_type: 'classification' or 'rating'
        answers: List of possible answers (default from DEFAULT_ANSWERS)
    """
    if answers is None:
        answers = DEFAULT_ANSWERS[response_type]
        print(f"INFO: Using default answers for '{response_type}': {answers}")

    try:
        results = pd.read_csv(os.path.join(results_dir, 'results_raw.csv'))
    except FileNotFoundError:
        return
    if 'Result' not in results.columns or results.empty:
        return

    results, column_names = parse_custom_ids(results)
    results['Result'] = pd.to_numeric(results['Result'], errors='coerce')
    groupby_cols = [c for c in column_names if c != 'trial']

    if response_type == 'classification':
        if 'trial' in column_names:
            out = results.groupby(groupby_cols)['Result'].mean().reset_index()
        else:
            out = results[groupby_cols + ['Result']].copy()
        out.rename(columns={'Result': 'prob_yes'}, inplace=True)

    else:
        # Compute frequency distribution over answer classes
        probs = (
            results.groupby(groupby_cols)['Result']
            .value_counts(normalize=True)
            .rename('prob')
            .reset_index()
        )
        # Ensure all answers are present for every query
        keys = probs[groupby_cols].drop_duplicates()
        complete = keys.merge(pd.DataFrame({'Result': answers}), how='cross')
        probs = complete.merge(probs, how='left').fillna({'prob': 0})

        out = _format_rating_output(probs, groupby_cols, 'Result', answers)

    out.to_csv(os.path.join(results_dir, 'probs.csv'), index=False)


# ---- Tensor logprobs → probabilities ----

def compute_binary_probabilities(input_dir: str, model_name: str, output_csv: str = None) -> dict:
    """
    Compute yes/no probabilities from saved logprob tensors (classification only).

    Processes .pt files saved by local_inference_logprob_tensors(), extracts the
    first token's probability distribution, and computes P(yes) / (P(yes) + P(no)).

    Note: This only supports binary classification tasks. For rating tasks,
    logprob tensors would need different post-processing (not yet implemented).

    Output: probs.csv with columns custom_id, prob_yes (standard format).

    Args:
        input_dir: Directory containing logprobs/ subdirectory with .pt files
        model_name: HuggingFace model name for tokenizer (e.g., 'allenai/OLMo-2-0325-32B')
        output_csv: Optional path for output CSV. If None, saves to input_dir/probs.csv

    Returns:
        Dictionary mapping custom_id to yes probability
    """
    from pathlib import Path
    import torch
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from .inference_local import find_yes_no_tokens, _build_token_mask

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    yes_token_ids, no_token_ids = find_yes_no_tokens(tokenizer)
    yes_token_ids = sorted(yes_token_ids)
    no_token_ids = sorted(no_token_ids)

    print(f"\nFound {len(yes_token_ids)} 'yes' token variants:")
    for token_id in yes_token_ids:
        print(f"  Token ID {token_id}: decoded={repr(tokenizer.decode([token_id]))}")

    print(f"\nFound {len(no_token_ids)} 'no' token variants:")
    for token_id in no_token_ids:
        print(f"  Token ID {token_id}: decoded={repr(tokenizer.decode([token_id]))}")

    input_path = Path(os.path.join(input_dir, "logprobs"))
    pt_files = sorted(input_path.glob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return {}

    print(f"\nFound {len(pt_files)} .pt files in {input_dir}")

    yes_mask = None
    no_mask = None
    other_mask = None
    expected_vocab_size = None

    yes_probs = {}
    skipped_files = []
    high_other_token_prob_count = 0

    for pt_file in tqdm(pt_files, desc="Processing files"):
        try:
            data = torch.load(pt_file)

            if isinstance(data, dict) and 'logprobs' in data:
                logprobs = data['logprobs']

                if len(logprobs.shape) != 2 or logprobs.shape[0] == 0:
                    print(f"Warning: Unexpected logprobs shape in {pt_file.name}: {logprobs.shape}")
                    skipped_files.append(pt_file.name)
                    continue

                first_token_logprobs = logprobs[0]
                vocab_size = logprobs.shape[1]

                if yes_mask is None:
                    expected_vocab_size = vocab_size
                    yes_mask = _build_token_mask(yes_token_ids, vocab_size, first_token_logprobs.device)
                    no_mask = _build_token_mask(no_token_ids, vocab_size, first_token_logprobs.device)
                    other_mask = ~(yes_mask | no_mask)
                elif vocab_size != expected_vocab_size:
                    print(f"Warning: Vocab size mismatch in {pt_file.name}: expected {expected_vocab_size}, got {vocab_size}")
                    skipped_files.append(pt_file.name)
                    continue

                all_first_token_probs = torch.softmax(first_token_logprobs, dim=0)

                yes_prob_first = all_first_token_probs[yes_mask].sum()
                no_prob_first = all_first_token_probs[no_mask].sum()
                other_prob_first = all_first_token_probs[other_mask].sum()

                if other_prob_first > 0.1 and logprobs.shape[0] > 1:
                    high_other_token_prob_count += 1
                    if high_other_token_prob_count < 10:
                        print(f"\n{pt_file.name}: High 'other' probability ({other_prob_first:.3f})")
                        print(f"  P(yes|first)={yes_prob_first:.3f}, P(no|first)={no_prob_first:.3f}")

                total = yes_prob_first + no_prob_first
                if total > 1e-10:
                    yes_probs[pt_file.name[:-3]] = (yes_prob_first / total).item()
                else:
                    skipped_files.append(pt_file.name)
                    print("Skipping file: ", pt_file.name)
            else:
                print(f"Warning: No 'logprobs' key found in {pt_file.name}")
                skipped_files.append(pt_file.name)

        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
            skipped_files.append(pt_file.name)

    if yes_probs:
        results_df = pd.DataFrame(list(yes_probs.items()), columns=["custom_id", "prob_yes"])
        if output_csv is None:
            output_csv = os.path.join(input_dir, "probs.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved probabilities to {output_csv}")

        if skipped_files:
            print(f"\nSkipped {len(skipped_files)} files:")
            for filename in skipped_files:
                print(f"  {filename}")

        print(f"\nQueries with significant probability mass on other tokens: {high_other_token_prob_count}/{len(yes_probs)}")

        yes_prob_values = list(yes_probs.values())
        print(f"\nStatistics:")
        print(f"  Number of files processed: {len(yes_probs)}")
        print(f"  Mean yes probability: {np.mean(yes_prob_values):.4f}")
        print(f"  Min yes probability: {min(yes_prob_values):.4f}")
        print(f"  Max yes probability: {max(yes_prob_values):.4f}")
        print(f"  std yes probability: {np.std(yes_prob_values):.4f}")
    else:
        print("No data was extracted")

    return yes_probs


# ---- Unified entry point ----

def postprocess(out_dir: str, response_type: str = None, answers: list = None) -> None:
    """
    Auto-detect source format from inference_settings.json and produce probs.csv.

    Reads the saved config to determine which processing function to call.
    Optional overrides allow re-running with different settings.

    Args:
        out_dir: Directory containing inference outputs and inference_settings.json
        response_type: Override response type (default: read from config)
        answers: Override answer set (default: from DEFAULT_ANSWERS)
    """
    config_path = os.path.join(out_dir, 'inference_settings.json')
    with open(config_path) as f:
        config = json.load(f)

    response_type = response_type or config.get('response_type')
    if response_type is None:
        raise ValueError(f"No response_type in {config_path} and none provided as override")

    if response_type == 'text':
        print("INFO: response_type is 'text', nothing to postprocess")
        return

    source = config.get('source', 'api')

    if source == 'local':
        print("INFO: Local inference already produced probs.csv directly, nothing to postprocess")
        return
    elif source == 'local_tensors':
        model_name = config.get('model_name')
        if model_name is None:
            raise ValueError(f"No model_name in {config_path}, needed for tensor postprocessing")
        compute_binary_probabilities(out_dir, model_name)
    elif config.get('get_logprobs'):
        logprobs_to_probabilities(out_dir, response_type=response_type, answers=answers)
    else:
        results_to_probabilities(out_dir, response_type=response_type, answers=answers)


# ---- CLI ----

def _postprocessing_cli():
    """CLI entry point for post-processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process inference outputs. Reads inference_settings.json to auto-detect the source format.",
        epilog="Example: python -m delta.inference.postprocessing --dir results/my_exp",
    )
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing inference outputs and inference_settings.json')
    parser.add_argument('--response-type', type=str, choices=['classification', 'rating'],
                        default=None, help='Override response type (default: read from config)')
    parser.add_argument('--answers', type=int, nargs='+', default=None,
                        help='Override answer set (default: auto from response type)')

    args = parser.parse_args()
    postprocess(args.dir, response_type=args.response_type, answers=args.answers)


if __name__ == "__main__":
    _postprocessing_cli()
