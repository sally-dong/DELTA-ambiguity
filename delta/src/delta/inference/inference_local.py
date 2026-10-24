import json
import math
import csv
import os
import time
import contextlib
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# =============================================================================
# Job Timing
# =============================================================================

@contextlib.contextmanager
def _job_timer(function_name: str):
    """Context manager that prints job start/end times and elapsed duration."""
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"JOB START: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"Function: {function_name}")
    print(f"{'='*80}\n")

    yield

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n{'='*80}")
    print(f"JOB END: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"{'='*80}\n")


# =============================================================================
# Yes/No Probability Helpers
# =============================================================================

def find_yes_no_tokens(tokenizer) -> Tuple[list[int], list[int]]:
    """
    Find all 'yes' and 'no' token IDs in the vocabulary.

    Note: We decode each token to get the actual text, since the raw vocabulary
    may contain special characters like Ġ (representing spaces in GPT-style tokenizers).
    """
    vocab = tokenizer.get_vocab()

    yes_token_ids = []
    no_token_ids = []

    for token, token_id in vocab.items():
        decoded = tokenizer.decode([token_id]).strip()
        decoded_lower = decoded.lower()

        if decoded_lower == "yes":
            yes_token_ids.append(token_id)
        elif decoded_lower == "no":
            no_token_ids.append(token_id)

    return yes_token_ids, no_token_ids


def _build_token_mask(token_ids: list[int], vocab_size: int, device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if token_ids:
        valid_ids = [tid for tid in token_ids if tid < vocab_size]
        if len(valid_ids) != len(token_ids):
            print("Warning: some yes/no token ids were outside vocab size; skipping those.")
        if valid_ids:
            mask[torch.tensor(valid_ids, device=device)] = True
    return mask


def _compute_yes_probability(logits: torch.Tensor, yes_mask: torch.Tensor, no_mask: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    yes_prob = probs[yes_mask].sum()
    no_prob = probs[no_mask].sum()
    total = yes_prob + no_prob
    if total <= 1e-12:
        return float("nan")
    return (yes_prob / total).item()


# =============================================================================
# Common Setup Helpers
# =============================================================================

def _get_device() -> str:
    """Detect and return the appropriate device (cuda or cpu) with warnings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: No GPU detected, running on CPU. This will be slow!")
    return device


def _load_tokenizer_and_model(model_name: str, device: str, quantize: bool = True):
    """Load model with appropriate settings based on device.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on ('cuda' or 'cpu')
        quantize: Whether to use 8-bit quantization (only applies to CUDA)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Important for decoder-only models in batched generation

    if device == "cuda":
        if quantize:
            print("Loading model with 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            print("Loading model without quantization (float16)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)

    model.eval()
    torch.set_grad_enabled(False)
    return tokenizer, model


@dataclass
class InferenceSetup:
    """Holds the result of inference setup."""
    device: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    query_inputs: list[dict]
    batch_size: int
    out_dir: str


def _setup_inference(
    inference_settings: dict,
    query_inputs: list[dict],
    output_file: str,
    use_csv: bool = False,
    response_type: str = None,
    source: str = "local",
) -> Optional[Tuple[InferenceSetup, str]]:
    """
    Common setup for all inference functions.

    Returns (InferenceSetup, output_path) or None if nothing to process.
    """
    from ._utils import init_output_dir, load_processed_ids
    from .batch_utils import filter_remaining_prompts

    print(f"Running model {inference_settings['model_name']}")
    print(f"Batch size: {inference_settings['batch_size']}")

    device = _get_device()

    out_dir = inference_settings['out_dir']
    config = {**inference_settings, 'response_type': response_type, 'source': source}
    config.pop('out_dir', None)
    init_output_dir(out_dir, config, check_existing=False)

    output_path = os.path.join(out_dir, output_file)
    file_type = 'csv' if use_csv else 'jsonl'
    processed_ids = load_processed_ids(output_path, file_type=file_type)

    filtered_inputs = filter_remaining_prompts(query_inputs, processed_ids)
    if len(filtered_inputs) == 0:
        print("All items already processed! Nothing to do.")
        return None

    quantize = inference_settings.get('quantize', True)
    tokenizer, model = _load_tokenizer_and_model(inference_settings['model_name'], device, quantize)

    setup = InferenceSetup(
        device=device,
        tokenizer=tokenizer,
        model=model,
        query_inputs=filtered_inputs,
        batch_size=inference_settings['batch_size'],
        out_dir=out_dir,
    )
    return setup, output_path


def _iter_batches(query_inputs: list[dict], batch_size: int):
    """Iterate over query_inputs in batches with progress bar."""
    num_batches = math.ceil(len(query_inputs) / batch_size)
    print(f"Starting batched inference on {len(query_inputs)} items")

    for batch_idx in tqdm(range(0, len(query_inputs), batch_size), total=num_batches, desc="Processing batches"):
        batch = query_inputs[batch_idx:batch_idx + batch_size]
        custom_ids = [q["custom_id"] for q in batch]
        prompts = [q["user_prompt"] for q in batch]
        yield batch, custom_ids, prompts


def _generate_batch(tokenizer, model, prompts: list[str], gen_kwargs: dict):
    """Tokenize prompts and run generation."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, use_cache=True, **gen_kwargs)
    return inputs, out


# =============================================================================
# Inference Functions
# =============================================================================

def local_inference_logprob_tensors(*, inference_settings: dict, query_inputs: list[dict]):
    """
    Run batched local inference and save full logprob tensors.

    Saves the complete logprob distribution for each generated token as .pt files.
    Post-process with postprocessing.compute_binary_probabilities() to get probs.csv.

    Note: compute_binary_probabilities() currently only supports classification
    (yes/no) tasks. For rating tasks, custom post-processing of the .pt files
    would be needed.

    Args:
        inference_settings: dict with 'model_name', 'out_dir', 'batch_size', 'max_output_tokens', and 'temperature'
        query_inputs: List of dicts with 'custom_id' and 'user_prompt' keys
    """
    with _job_timer("local_inference_with_logprob_tensor"):
        result = _setup_inference(inference_settings, query_inputs, "manifest.jsonl",
                                  response_type="classification", source="local_tensors")
        if result is None:
            return
        setup, manifest_path = result

        os.makedirs(os.path.join(setup.out_dir, "logprobs"), exist_ok=True)

        gen_kwargs = {
            "max_new_tokens": inference_settings['max_output_tokens'],
            "temperature": inference_settings['temperature'],
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": inference_settings['temperature'] > 0.0,
            "pad_token_id": setup.model.config.eos_token_id,
            "eos_token_id": setup.model.config.eos_token_id,
        }

        if hasattr(setup.model, 'hf_device_map'):
            print(setup.model.hf_device_map)

        with open(manifest_path, "a", encoding="utf-8") as mf:
            for batch, custom_ids, prompts in _iter_batches(setup.query_inputs, setup.batch_size):
                inputs, out = _generate_batch(setup.tokenizer, setup.model, prompts, gen_kwargs)

                for idx in range(len(batch)):
                    custom_id = custom_ids[idx]
                    seq = out.sequences[idx]
                    input_ids = inputs["input_ids"][idx]
                    gen_ids = seq[len(input_ids):]
                    completion = setup.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    # Build full-distribution logprobs tensor [T, V]
                    logprob_steps = [
                        torch.log_softmax(logits_step[idx], dim=-1)
                        for logits_step in out.scores
                    ]
                    logprobs = torch.stack(logprob_steps, dim=0).to(dtype=torch.float16)

                    token_ids = gen_ids.detach().cpu()
                    tokens = setup.tokenizer.convert_ids_to_tokens(token_ids.tolist())

                    payload = {
                        "custom_id": custom_id,
                        "completion": completion,
                        "steps": int(logprobs.shape[0]),
                        "token_ids": token_ids,
                        "tokens": tokens,
                        "logprobs": logprobs.cpu(),
                    }
                    torch.save(payload, os.path.join(setup.out_dir, f"logprobs/{custom_id}.pt"))

                    json.dump({"custom_id": custom_id, "completion": completion}, mf)
                    mf.write("\n")

        print(f"Wrote manifest to {manifest_path}")


def local_inference_binary_probabilities(*, inference_settings: dict, query_inputs: list[dict]) -> None:
    """
    Run batched local inference and write yes/no probabilities to CSV.

    Args:
        inference_settings: Dict with keys:
            - model_name (str): HuggingFace model identifier
            - out_dir (str): Output directory
            - batch_size (int): Batch size for processing
            - quantize (bool, optional): Use 8-bit quantization (default True)
        query_inputs: List of dicts with 'custom_id' and 'user_prompt' keys
    """
    with _job_timer("local_inference_with_probabilities"):
        if inference_settings.get('max_output_tokens', 1) > 1:
            print("Warning: max_output_tokens > 1; only the first token's probabilities will be computed.")

        result = _setup_inference(inference_settings, query_inputs, "probs.csv", use_csv=True,
                                  response_type="classification", source="local")
        if result is None:
            return
        setup, probs_csv_path = result

        yes_token_ids, no_token_ids = find_yes_no_tokens(setup.tokenizer)
        print(f"Found {len(yes_token_ids)} yes tokens and {len(no_token_ids)} no tokens")

        gen_kwargs = {
            "max_new_tokens": 1,
            "temperature": 0,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": setup.model.config.eos_token_id,
            "eos_token_id": setup.model.config.eos_token_id,
        }

        vocab_size = setup.model.config.vocab_size
        yes_mask = _build_token_mask(yes_token_ids, vocab_size, setup.device).to(setup.model.device)
        no_mask = _build_token_mask(no_token_ids, vocab_size, setup.device).to(setup.model.device)

        csv_exists = os.path.exists(probs_csv_path)

        with open(probs_csv_path, "a", encoding="utf-8", newline="") as pf:
            writer = csv.DictWriter(pf, fieldnames=["custom_id", "prob_yes"])
            if not csv_exists:
                writer.writeheader()
                pf.flush()

            for batch, custom_ids, prompts in _iter_batches(setup.query_inputs, setup.batch_size):
                _, out = _generate_batch(setup.tokenizer, setup.model, prompts, gen_kwargs)

                for idx in range(len(batch)):
                    custom_id = custom_ids[idx]

                    if not out.scores:
                        print(f"Warning: No scores returned for {custom_id}; skipping probability computation.")
                        prob_yes = float("nan")
                    else:
                        logits_first_token = out.scores[0][idx]
                        if logits_first_token.shape[-1] != vocab_size:
                            print(f"Warning: vocab size mismatch for {custom_id}; expected {vocab_size}, "
                                  f"got {logits_first_token.shape[-1]}. Skipping.")
                            prob_yes = float("nan")
                        else:
                            prob_yes = _compute_yes_probability(
                                logits_first_token,
                                yes_mask.to(logits_first_token.device),
                                no_mask.to(logits_first_token.device)
                            )

                    writer.writerow({"custom_id": custom_id, "prob_yes": prob_yes})

                pf.flush()

        print(f"Wrote probabilities to {probs_csv_path}")


def local_inference_simple(*, inference_settings: dict, query_inputs: list[dict]) -> None:
    """
    Run batched local inference and write responses to results.csv.

    Args:
        inference_settings: dict with 'model_name', 'out_dir', 'max_output_tokens', 'batch_size', and 'temperature'
        query_inputs: List of dicts with 'custom_id' and 'user_prompt' keys
    """
    with _job_timer("local_inference_simple"):
        print(f"Max output tokens: {inference_settings['max_output_tokens']}")

        result = _setup_inference(inference_settings, query_inputs, "results.csv", use_csv=True,
                                  response_type="text", source="local")
        if result is None:
            return
        setup, results_path = result

        gen_kwargs = {
            "max_new_tokens": inference_settings['max_output_tokens'],
            "temperature": inference_settings['temperature'],
            "do_sample": inference_settings['temperature'] > 0.0,
            "pad_token_id": setup.model.config.eos_token_id,
            "eos_token_id": setup.model.config.eos_token_id,
        }

        csv_exists = os.path.exists(results_path) and os.path.getsize(results_path) > 0

        with open(results_path, "a", encoding="utf-8", newline="") as rf:
            writer = csv.DictWriter(rf, fieldnames=["custom_id", "result"])
            if not csv_exists:
                writer.writeheader()
                rf.flush()

            for batch, custom_ids, prompts in _iter_batches(setup.query_inputs, setup.batch_size):
                inputs, out = _generate_batch(setup.tokenizer, setup.model, prompts, gen_kwargs)

                for idx in range(len(batch)):
                    custom_id = custom_ids[idx]
                    seq = out[idx]
                    input_ids = inputs["input_ids"][idx]
                    gen_ids = seq[len(input_ids):]
                    response = setup.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    writer.writerow({"custom_id": custom_id, "result": response})

                rf.flush()

        print(f"Wrote results to {results_path}")


