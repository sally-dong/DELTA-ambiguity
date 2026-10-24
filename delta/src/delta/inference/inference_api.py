"""
Streaming inference functions for real-time LLM queries.
"""

import os
import csv
import re
import json
import pprint
import requests
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from google.genai import types
from tqdm import tqdm

from .config import Config


def clean_response_text(response_type: str, response_text: str):
    """
    Clean the response text to a single number or string.

    Args:
        response_type: 'rating', 'classification', or 'text'
        response_text: the text returned by the LLM

    Returns:
        Cleaned response text, or None if invalid
    """
    if not response_text:
        return None

    if response_type == 'rating':
        match = re.search(r'\d', response_text)
        if match:
            response_text = match.group()
        else:
            return None

    elif response_type == 'classification':
        if 'yes' in response_text.lower():
            response_text = '1'
        elif 'no' in response_text.lower():
            response_text = '0'
        else:
            return None

    return response_text

def _query_with_retry(query_fn, inp: dict, max_retries: int = 3, base_delay: float = 10.0):
    """
    Wrap a query function with exponential backoff retry logic.

    Args:
        query_fn: The query function to call
        inp: The input to pass to the query function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)

    Returns:
        Tuple of (response, error_message). On success, error_message is None.
        On failure after all retries, response is None and error_message contains the last error.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = query_fn(inp)
            return response, None
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    return None, str(last_error)

def _query_model(inference_settings: dict, inp: dict):
    """
    Make an API call to the specified model.

    This is an internal helper function that handles the differences between
    Gemini, OpenAI, and OpenRouter APIs.

    Args:
        inference_settings: Dictionary containing model configuration
        inp: Dictionary with 'system_prompt' and 'user_prompt'

    Returns:
        API response object (varies by model type)
    """
    if 'gemini' in inference_settings['model']:
        client = Config.GEMINI_CLIENT()
        if inference_settings['thinking']:
            return client.models.generate_content(
                model=inference_settings['model'],
                config=types.GenerateContentConfig(
                    system_instruction=inp['system_prompt'],
                    temperature=inference_settings['temperature'],
                    thinking_config=types.ThinkingConfig(thinking_budget=inference_settings['thinking_budget'])
                ),
                contents=inp['user_prompt'])
        else:
            logprobs = None
            if inference_settings['get_logprobs']:
                logprobs = inference_settings['k_logprobs']
            return client.models.generate_content(
                model=inference_settings['model'],
                config=types.GenerateContentConfig(
                    system_instruction=inp['system_prompt'],
                    temperature=inference_settings['temperature'],
                    response_logprobs=inference_settings['get_logprobs'],
                    logprobs=logprobs,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
                contents=inp['user_prompt'])

    elif 'gpt' in inference_settings['model']:
        client = Config.OPENAI_CLIENT()
        messages = [
            {"role": "system", "content": inp['system_prompt']},
            {"role": "user", "content": inp['user_prompt']},
        ]
        kwargs = dict(
            model=inference_settings['model'],
            input=messages,
            service_tier="flex",
        )
        if inference_settings['thinking']:
            kwargs['reasoning'] = {"effort": inference_settings['thinking_effort']}
            kwargs['max_output_tokens'] = inference_settings['max_output_tokens']
        else:
            kwargs['temperature'] = inference_settings['temperature']
            if inference_settings['get_logprobs']:
                kwargs['top_logprobs'] = inference_settings['k_logprobs']
                kwargs['include'] = ["message.output_text.logprobs"]
        return client.responses.create(**kwargs)

    # OpenRouter
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY()}",
        },
        data=json.dumps({
            "model": inference_settings['model'],
            "temperature": inference_settings['temperature'],
            "messages": [
                {
                    "role": "system",
                    "content": inp['system_prompt']
                },
                {
                    "role": "user",
                    "content": inp['user_prompt']
                }],
            "logprobs": True,
            "top_logprobs": inference_settings['k_logprobs'],
        }))
    return response


def _extract_response(inference_settings: dict, response) -> tuple[str, list[tuple[str, float]]]:
    """
    Extract response text and logprobs from an API response.

    Handles Gemini, GPT/OpenAI, and OpenRouter response formats.

    Returns:
        (response_text, token_logprobs) where token_logprobs is empty if logprobs were not requested.
        Returns (None, []) if the model type is unrecognized.
    """
    token_logprobs = []

    if 'gemini' in inference_settings['model']:
        response_text = response.text
        if inference_settings.get('get_logprobs'):
            token_logprobs = get_gemini_logprobs(response)

    elif 'gpt' in inference_settings['model']:
        response_text = response.output_text
        if inference_settings.get('get_logprobs'):
            token_logprobs = get_gpt_logprobs(response)

    elif inference_settings.get('is_openrouter'):
        data = response.json()
        response_text = data["choices"][0]["message"]["content"]

    else:
        return None, []

    return response_text, token_logprobs


def get_gemini_logprobs(response) -> list[tuple[str, float]]:
    """
    Extract log probabilities from a Gemini response.

    Returns:
        List of (token, log_probability) tuples for the first generated token
    """
    token_logprobs = []
    if response.candidates and response.candidates[0].logprobs_result:
        logprobs_result = response.candidates[0].logprobs_result
        top_alternatives = logprobs_result.top_candidates[0].candidates
        if top_alternatives:
            for alt_token_info in top_alternatives:
                token_logprobs.append((alt_token_info.token, alt_token_info.log_probability))
    return token_logprobs


def get_gpt_logprobs(response) -> list[tuple[str, float]]:
    """
    Extract log probabilities from an OpenAI response.

    Returns:
        List of (token, log_probability) tuples for the first generated token
    """
    token_logprobs = []
    logprob_list = response.output[0].content[0].logprobs[0].top_logprobs
    if logprob_list:
        for logprob_obj in logprob_list:
            token = logprob_obj.token
            logprob_value = logprob_obj.logprob
            token_logprobs.append((token, logprob_value))
    return token_logprobs


def query(inp: dict, inference_settings: dict, verbose: bool = True):
    """
    Make a single API query to the model of choice.

    Args:
        inp: Dictionary with 'custom_id', 'system_prompt', 'user_prompt'
        inference_settings: Model and inference configuration dictionary
        verbose: If True, print response and logprobs

    Returns:
        Response text from the model
    """
    response = _query_model(inference_settings, inp)
    response_text, token_logprobs = _extract_response(inference_settings, response)

    if response_text is None:
        print(f"Parsing failed for {inp.get('custom_id', '?')}: original response: {response.__dict__}")
        return None

    if verbose:
        print("Model response:\n", response_text)
        if token_logprobs:
            print("logprobs:")
            for token, logprob_value in token_logprobs:
                print(token, logprob_value)

    return response_text


def run_inference(*, out_dir: str, query_inputs: list[dict], inference_settings: dict, response_type: str):
    """
    Run all queries concurrently against the chosen LLM using multiple threads.

    Answers are immediately written to file. Can be rerun without repeating
    already-completed queries (idempotent).

    Args:
        out_dir: Directory where all data and results are saved
        query_inputs: List of dictionaries, each containing:
            - custom_id: a unique identifier for the input
            - system_prompt: the system prompt for the input
            - user_prompt: the user prompt for the input
        inference_settings: Dict with keys:
            - model (str): Model identifier (e.g. 'gpt-4.1-mini', 'gemini-2.5-flash')
            - temperature (float): Sampling temperature
            - thinking (bool): Enable extended thinking mode
            - thinking_effort (str, optional): OpenAI thinking effort ('low'/'medium'/'high')
            - thinking_budget (int, optional): Gemini thinking token budget
            - max_output_tokens (int, optional): Maximum completion length
            - is_openrouter (bool, optional): Route through OpenRouter
            - get_logprobs (bool, optional): Extract logprobs from response
            - k_logprobs (int, optional): Number of top logprobs to return
        response_type: One of 'rating' (1 numerical digit), 'classification' (yes/no), or 'text' (freeform)
    """
    assert response_type in ['rating', 'classification', 'text'], f"Invalid response type: {response_type}"
    pprint.pprint(inference_settings)

    from ._utils import init_output_dir, load_processed_ids
    from .batch_utils import filter_remaining_prompts

    config = {**inference_settings, 'response_type': response_type, 'source': 'api'}
    init_output_dir(out_dir, config, check_existing=True)

    # Load existing results or initialize output files
    results_path = os.path.join(out_dir, 'results_raw.csv')
    processed_ids = load_processed_ids(results_path, file_type='csv')
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            csv.writer(f).writerow(['custom_id', 'Result'])
        with open(os.path.join(out_dir, 'logprobs_raw.csv'), 'w') as f:
            csv.writer(f).writerow(['custom_id', 'token', 'logprob'])

    remaining_inputs = filter_remaining_prompts(query_inputs, processed_ids)
    total_remaining = len(remaining_inputs)
    print(f"Processed {len(processed_ids)} entries, {total_remaining} remaining")

    file_lock = Lock()
    logprob_file_lock = Lock()
    query_model_fn = partial(_query_model, inference_settings)
    query_with_retry_fn = partial(_query_with_retry, query_model_fn)
    clean_response_fn = partial(clean_response_text, response_type)

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_input_dict = {
            executor.submit(query_with_retry_fn, inp): inp
            for inp in remaining_inputs
        }

        with open(os.path.join(out_dir, 'results_raw.csv'), 'a', newline='') as result_file, \
                open(os.path.join(out_dir, 'logprobs_raw.csv'), 'a', newline='') as logprob_file:

            writer = csv.writer(result_file)
            logprob_writer = csv.writer(logprob_file)

            for future in tqdm(as_completed(future_to_input_dict), total=total_remaining, desc="Processing queries"):
                inp = future_to_input_dict[future]
                response, error = future.result()
                if error is not None:
                    print(f"Failed after retries for {inp['custom_id']}: {error}")
                    continue

                response_text, token_logprobs = _extract_response(inference_settings, response)
                if response_text is None:
                    print(f"Unknown model type for {inp['custom_id']}: {inference_settings['model']}")
                    continue

                response_text = clean_response_fn(response_text)
                if response_text is None:
                    print(f"Parsing failed for {inp['custom_id']}")
                    continue

                with file_lock:
                    writer.writerow([inp['custom_id'], response_text])
                    result_file.flush()

                if token_logprobs:
                    with logprob_file_lock:
                        for token, logprob_value in token_logprobs:
                            logprob_writer.writerow([inp['custom_id'], token, logprob_value])
                        logprob_file.flush()

    # Auto-produce probs.csv for classification/rating tasks
    from .postprocessing import postprocess
    postprocess(out_dir)
