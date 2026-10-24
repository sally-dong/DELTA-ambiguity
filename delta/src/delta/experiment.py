import json
import math
import argparse
import textwrap
import inspect
import os
from typing import Callable

from delta.inference import (
    run_inference,
    local_inference_binary_probabilities,
    add_batch_args,
    get_batch,
    get_batch_output_dir,
    load_completed_ids_from_batches,
    filter_remaining_prompts,
    Config,
)

# API keys are read from OPENAI_API_KEY / GEMINI_API_KEY by default. To use
# different variable names, call Config.set_api_key_env_vars() before querying.
Config.set_max_workers(128)

from delta.inference.batch_utils import get_slurm_batch_index, get_next_batch_index
from delta.prompts import get_prompt_fn

def format_all_prompts(*, out_dir:str, raters:list[str], policies:list[dict], cases:list[str], prompt_fn:Callable, scenario_params:dict, instruct_tuned:bool, system_prompt:str=None, n_trials:int=1) -> list[dict]:
    """
    Build all (rater x policy x case x prompt_variant x trial) prompt combinations.

    Args:
        system_prompt: Override the system prompt. If None, uses a default yes/no
                       instruction for instruct-tuned models, or empty string for base models.
    """
    prompts = []
    if system_prompt is None:
        system_prompt = "Answer ONLY 'yes' or 'no' only to the question below. Do not provide an explanation." if instruct_tuned else ""

    for index_rater, rater in enumerate(raters):
        for index_policy, policy_dict in enumerate(policies):
            policy = policy_dict['text']
            for index_case, case in enumerate(cases):
                prompt_variants = prompt_fn(rater=rater, policy=policy, case=case, scenario_params=scenario_params, instruct_tuned=instruct_tuned)
                for prompt_index, prompt in enumerate(prompt_variants):
                    if n_trials > 1:
                        for trial in range(n_trials):
                            custom_id = f'rater-{index_rater}_policy-{index_policy}_case-{index_case}_prompt-{prompt_index}_trial-{trial}'
                            prompts.append({"custom_id": custom_id, "system_prompt": system_prompt, "user_prompt": prompt})
                    else:
                        custom_id = f'rater-{index_rater}_policy-{index_policy}_case-{index_case}_prompt-{prompt_index}'
                        prompts.append({"custom_id": custom_id, "system_prompt": system_prompt, "user_prompt": prompt})

    print("Sample prompt: ", prompts[0])
    print("Total prompts: ", len(prompts))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'input_construction.json'), 'w') as file:
        format_fn_text = textwrap.dedent(inspect.getsource(prompt_fn))
        sample_query = prompts[0]
        json.dump({'sample_query':sample_query, 'prompt_construction':format_fn_text, 'raters':raters, 'policies':policies, 'cases':cases, 'n_trials':n_trials}, file, indent=2, ensure_ascii=False)

    return prompts

def make_local_inference_settings(model_name, out_dir):
    return {
        'model_name': model_name,
        'max_output_tokens': 1,
        'out_dir': out_dir,
        'batch_size': 16,
        'temperature': 0.0,
        'quantize': True,
    }

def run_local_batched_inference(prompts, base_output_dir, model_name, args):
    """Handle batching, completion tracking, and local inference."""

    total_batches = math.ceil(len(prompts) / args.batch_size)

    # 1. Get batch index from SLURM_ARRAY_TASK_ID (or 0 if not in array job)
    batch_index = get_slurm_batch_index()

    # 2. Check if user just wants to know total batches
    if args.total_batches:
        print(f"\n{'='*60}")
        print(f"BATCH CONFIGURATION")
        print(f"{'='*60}")
        print(f"Total prompts:      {len(prompts):,}")
        print(f"Batch size:         {args.batch_size:,}")
        print(f"Total batches:      {total_batches}")
        print(f"\nTo run all batches:")
        print(f"  sbatch --array=0-{total_batches-1} --export=MODEL={args.model},DIR={args.dir} slurm/run_job_batched.sbatch")
        print(f"{'='*60}\n")
        exit(0)

    # 3. Alternative: Run all remaining prompts at once
    if args.run_remaining:
        completed_ids = load_completed_ids_from_batches(base_output_dir, file_type="csv")
        remaining_prompts = filter_remaining_prompts(prompts, completed_ids)

        print(f"\n{'='*60}")
        print(f"RUN REMAINING MODE. WARNING: DO NOT RESTART REGULAR BATCHING")
        print(f"{'='*60}")
        print(f"Total prompts:      {len(prompts):,}")
        print(f"Already completed:  {len(completed_ids):,}")
        print(f"Remaining:          {len(remaining_prompts):,}")

        if not remaining_prompts:
            print(f"\nAll prompts are already complete. Exiting successfully.")
            print(f"{'='*60}\n")
            exit(0)

        next_batch_idx = get_next_batch_index(base_output_dir)
        out_dir = get_batch_output_dir(base_output_dir, next_batch_idx)
        print(f"Output directory:   {out_dir}")
        print(f"{'='*60}\n")

        settings = make_local_inference_settings(model_name, out_dir)
        local_inference_binary_probabilities(inference_settings=settings, query_inputs=remaining_prompts)
        exit(0)

    # 4. Get prompts for this batch (deterministic assignment)
    batch_prompts = get_batch(prompts, args.batch_size, batch_index)
    print(f"Batch {batch_index}: {len(batch_prompts):,} prompts (indices {batch_index * args.batch_size} to {batch_index * args.batch_size + len(batch_prompts) - 1})")

    # 5. Load completed IDs for this batch and filter
    out_dir = get_batch_output_dir(base_output_dir, batch_index)
    completed_ids = load_completed_ids_from_batches(base_output_dir, file_type="csv")
    remaining_prompts = filter_remaining_prompts(batch_prompts, completed_ids)

    print(f"Already completed in batch {batch_index}: {len(batch_prompts) - len(remaining_prompts):,}")
    print(f"Remaining to process: {len(remaining_prompts):,}")

    if not remaining_prompts:
        print(f"\nBatch {batch_index} is already complete. Exiting successfully.")
        exit(0)

    settings = make_local_inference_settings(model_name, out_dir)
    local_inference_binary_probabilities(inference_settings=settings, query_inputs=remaining_prompts)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to use, options are 'olm', 'qwen', 'gpt', 'gemini'")
    parser.add_argument("--dir", required=True,
                        help="experiment directory holding policies.json, "
                             "cases.json, longest_raters.json and "
                             "scenario_params.json")
    parser.add_argument("--out-dir", default=None,
                        help="where to write results (default: "
                             "results/{basename of --dir}_{model})")
    parser.add_argument("--suffix", required=False, default="", help="optional suffix for output directory")
    parser.add_argument("--policies-file", default="policies.json",
                        help="policies file under --dir (default: policies.json)")
    parser.add_argument("--cases-suffix", default="",
                        help="suffix on input cases filenames, e.g. '_2' -> sampled_cases_2.json / cases_2.json")
    parser = add_batch_args(parser)
    args = parser.parse_args()

    instruct_tuned = False
    n_trials = 1
    if args.model == 'olm':
        model_name = "allenai/OLMo-2-0325-32B"
    elif args.model == 'qwen':
        model_name = "Qwen/Qwen3-30B-A3B-Base"
    elif args.model == 'gpt':
        model_name = "gpt"
        instruct_tuned = True
        gen_kwargs = {
            'model': 'gpt-4.1-mini',
            'is_openrouter': False,
            'get_logprobs': False,
            'k_logprobs': 5,
            'temperature': 0,
            'thinking': False,
            'thinking_effort': "minimal",
            'max_output_tokens': 4,
        }
    elif args.model == 'gemini':
        model_name = "gemini"
        instruct_tuned = True
        gen_kwargs = {
            'model': 'gemini-2.5-flash',
            'is_openrouter': False,
            'get_logprobs': False,
            'k_logprobs': 5,
            'temperature': 1,
            'thinking': False,
            'thinking_budget': 0,
        }
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    dir = args.dir
    input_dir = dir
    if args.out_dir:
        output_dir = args.out_dir
    else:
        name = os.path.basename(os.path.normpath(dir))
        parts = [name, args.suffix, args.model] if args.suffix else [name, args.model]
        output_dir = os.path.join("results", "_".join(parts))

    with open(f"{input_dir}/{args.policies_file}", "r", encoding="utf-8") as f:
        policies = json.load(f)

    cases_path = f"{input_dir}/sampled_cases{args.cases_suffix}.json"
    if not os.path.exists(cases_path):
        cases_path = f"{input_dir}/cases{args.cases_suffix}.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    with open(f"{input_dir}/longest_raters.json", "r", encoding="utf-8") as f:
        raters = json.load(f)

    with open(f"{input_dir}/scenario_params.json", "r", encoding="utf-8") as f:
        scenario_params = json.load(f)

    prompts = format_all_prompts(out_dir=output_dir, raters=raters, policies=policies, cases=cases, prompt_fn=get_prompt_fn(scenario_params, input_dir), scenario_params=scenario_params, instruct_tuned=instruct_tuned, n_trials=n_trials)

    if model_name == "gpt" or model_name == "gemini":
        run_inference(out_dir=output_dir, query_inputs=prompts, inference_settings=gen_kwargs, response_type="classification")
    else:
        run_local_batched_inference(prompts, output_dir, model_name, args)


if __name__ == "__main__":
    main()
