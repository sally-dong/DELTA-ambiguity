import json
import argparse
import time
import numpy as np
from delta.inference import query


def _query_with_retry(query_input, inference_settings, max_retries=5, base_delay=30):
    """Wrap query() with exponential-backoff retry on transient errors (e.g. 429 rate limits)."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return query(query_input, inference_settings, verbose=False)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"  query failed ({type(e).__name__}: {e}); retrying in {delay}s...")
                time.sleep(delay)
    raise last_err


# Ordered policy pairs sampled to seed case generation, when
# scenario_params does not say. With the usual 3 cases per pair this yields
# 60 cases, around where DELTA score rankings stabilise in practice.
DEFAULT_N_POLICY_PAIRS = 20


def generate_cases(*, input_dir, policies, scenario_params, inference_settings, suffix='', **_):
    cases = []

    # Seed case generation from the top N policies (assumes the seed file is
    # sorted by quality/length). Raters still evaluate against the full policies
    # list at experiment time.
    n_top = scenario_params.get('n_top_policies_for_cases')
    case_policies = policies[:n_top] if n_top is not None else policies
    n = len(case_policies)
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Sample pairs rather than enumerating them. The full set grows as n(n-1),
    # so 30 policies would be 870 generation calls; set n_policy_pairs to null
    # to opt into all of them.
    n_policy_pairs = scenario_params.get('n_policy_pairs', DEFAULT_N_POLICY_PAIRS)
    if n_policy_pairs is not None and n_policy_pairs < len(all_pairs):
        seed = scenario_params.get('pair_sampling_seed', 42)
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(all_pairs), size=n_policy_pairs, replace=False).tolist())
        pairs = [all_pairs[i] for i in chosen]
    else:
        pairs = all_pairs

    out_path = f'{input_dir}/cases{suffix}.json'
    print(f"Generating cases for {len(pairs)} ordered pairs (of {len(all_pairs)} possible, "
          f"seeded from top {n} of {len(policies)} policies). Writing progressively to {out_path}")
    if 'n_policy_pairs' not in scenario_params and len(all_pairs) > DEFAULT_N_POLICY_PAIRS:
        print(f"  (n_policy_pairs unset, defaulting to {DEFAULT_N_POLICY_PAIRS}; "
              f"set it explicitly, or to null for all {len(all_pairs)} pairs)")

    for (i, j) in pairs:
        policy1 = case_policies[i]['text']
        policy2 = case_policies[j]['text']

        query_input = {
            "user_prompt":(f"Here are two versions of {scenario_params['policy_topic']}.\n"
                        f"Version 1: {policy1}\n"
                        f"Version 2: {policy2}\n"
                        f"Please come up with {scenario_params['case_per_ordered_policy_pair']} short, fictional scenarios that comply with the first version but violate the second."
                        f"Each scenario should be a {scenario_params['case_format_description']}. "
                        "They should be realistic and detailed enough to clearly illustrate the difference between the two statutes. "
                        "They should also be creative and cover a diverse range of situations.\n"
                        f"{scenario_params['case_example']}"
                        "Only output the scenarios, do not include any analysis, reasoning, or commentary."),
            "system_prompt":"Give your output as a JSON list of strings, where each string is one scenario.",
        }
        output = _query_with_retry(query_input, inference_settings)
        print(f"Generated cases for policy pair ({i}, {j})")
        try:
            output = json.loads(output)
            if len(output) != scenario_params['case_per_ordered_policy_pair']:
                print(f"warning: did not generate the right number of cases. Expected {scenario_params['case_per_ordered_policy_pair']}, got {len(output)}")
            for case in output:
                cases.append(case)
        except json.JSONDecodeError:
            print("Failed to decode JSON output:", output)
            break

        # Progressive save so we can watch the file grow and don't lose work on crash
        with open(out_path, 'w', encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)


def generate_base_raters(*, input_dir, scenario_params, inference_settings, **_):
    base_raters = []
    for _ in range(scenario_params['rater_generation_iterations']):
        query_input = {
            "user_prompt":(f"You are writing {scenario_params['policy_topic']} and interested in considering the perspectives of various stakeholders. "
                        f"Please make a list of {scenario_params['n_raters']} total stakeholders. Generate these stakeholders in pairs who have opposing interests in some aspect.\n"
                        "Here is an example pair:\n"
                        f"{scenario_params['rater_pair_example']}\n"
                        "Be creative in generating the stakeholders and draw from diverse professional identities while keeping the descriptions succinct. "
                        "Flatten the final output as a list of stakeholders without grouping them into pairs."),
            "system_prompt":"Give your output as a JSON list of strings, where each string is one stakeholder, beginning with the phrase 'This person is a...'.",
        }
        output = query(query_input, inference_settings, verbose=False)
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            print("Failed to decode JSON output:", output)
            break

        for rater in output:
            base_raters.append(rater)

    with open(f'{input_dir}/base_raters.json', 'w', encoding="utf-8") as f:
        json.dump(base_raters, f, ensure_ascii=False, indent=2)


def generate_expanded_raters(*, input_dir, scenario_params, inference_settings, **_):
    with open(f"{input_dir}/base_raters.json", "r", encoding="utf-8") as f:
        base_raters = json.load(f)

    expanded_raters = [] # a list of lists
    longest_raters = [] # a list of strings
    for rater in base_raters:
        rater_descriptions = [rater]
        for _ in range(1, scenario_params['rater_description_length']):
            query_input = {
                "user_prompt":(f"Here is a brief description of a person who is a stakeholder regarding {scenario_params['policy_topic']}.\n"
                            f"Description: {rater_descriptions[-1]}\n"
                            "Please write a very short sentence expanding on the description in connection with the policy topic. Be realistic and consistent about their position on the issue."),
                "system_prompt":"",
            }
            output = query(query_input, inference_settings, verbose=False)
            rater_descriptions.append(rater_descriptions[-1] + " " + output)

        expanded_raters.append(rater_descriptions)
        longest_raters.append(rater_descriptions[-1])

    with open(f'{input_dir}/expanded_raters.json', 'w', encoding="utf-8") as f:
        json.dump(expanded_raters, f, ensure_ascii=False, indent=2)

    with open(f'{input_dir}/longest_raters.json', 'w', encoding="utf-8") as f:
        json.dump(longest_raters, f, ensure_ascii=False, indent=2)

# N.B: this function is not used in the pipeline. It uses raters' priors to generate cases, rather than cases from scratch
def generate_cases_from_raters(*, input_dir, policies, scenario_params, inference_settings):
    with open(f"{input_dir}/longest_raters.json", "r", encoding="utf-8") as f:
        raters = json.load(f)

    cases = []
    for i, rater in enumerate(raters):
        for policy_dict in policies:
            policy = policy_dict['text']
            coin_toss = np.random.randint(2)
            issue = "too permissive" if coin_toss else "insufficiently strict"
            query_input = {
                    "user_prompt":(f"{rater}"
                                   f"This person is redteaming the following policy on {scenario_params['policy_topic']}."
                                   f"Policy: {policy}"
                                   f"Based on their own life experiences, they identify an issue in the policy text where it is {issue} as written than the spirit. "
                                   "Next, they come up with a fictional scenario involving a student's use of an AI, "
                                   "which clearly illustrates the issue with applying the policy as written."
                                   "The fictional scenario is as follows: "
                    ),
                    "system_prompt":"Stay in character. Output only the required scenario without any explanations or justifications.",
                }
            output = query(query_input, inference_settings, verbose=False)
            cases.append(output)
        print(f"Generated cases by rater {i}")

    with open(f'{input_dir}/rater_generated_cases.json', 'w', encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)


def sample_cases(*, input_dir, scenario_params, suffix='', **_):
    """Downsample cases{suffix}.json -> sampled_cases{suffix}.json.

    Reads scenario_params['n_sampled_cases'] and optional 'sampling_seed' (default 42).
    """
    with open(f"{input_dir}/cases{suffix}.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    n = scenario_params['n_sampled_cases']
    seed = scenario_params.get('sampling_seed', 42)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(cases), size=n, replace=False).tolist())
    sampled = [cases[i] for i in idx]

    with open(f"{input_dir}/sampled_cases{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"Sampled {n}/{len(cases)} cases (seed={seed}) -> sampled_cases{suffix}.json")


ACTIONS = {
    "cases": generate_cases,
    "base_raters": generate_base_raters,
    "expanded_raters": generate_expanded_raters,
    # "cases_from_raters": generate_cases_from_raters,
    "sample_cases": sample_cases,
}

BASE_INFERENCE_SETTINGS = {
    'model': 'gpt-5',
    'is_openrouter': False,
    'get_logprobs': False,
    'k_logprobs': 0,
    'thinking': True,
    'temperature': 1,
    'thinking_effort': "medium",
    'max_output_tokens': 10000,
}

# Per-action overrides applied on top of BASE_INFERENCE_SETTINGS.
# expanded_raters appends one short sentence per call (~50 tokens of useful
# output) and runs many times per experiment, so reasoning is wasted budget.
PER_ACTION_SETTINGS = {
    "expanded_raters": {
        'thinking': False,
        'thinking_effort': "minimal",
        'max_output_tokens': 200,
    },
}


def settings_for(action_name):
    return {**BASE_INFERENCE_SETTINGS, **PER_ACTION_SETTINGS.get(action_name, {})}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="experiment directory holding policies.json and "
                             "scenario_params.json")
    parser.add_argument("--action", required=True, choices=[*ACTIONS, "all"],
                        help="which generation step to run")
    parser.add_argument("--policies-file", default="policies.json",
                        help="policies file under --dir (default: policies.json)")
    parser.add_argument("--suffix", default="",
                        help="suffix for output filenames, e.g. '_2' -> cases_2.json")
    parser.add_argument("--seed-pool-size", type=int, default=None,
                        help="override scenario_params['n_top_policies_for_cases'] for this run; "
                             "use 0 to disable the top-N filter and use the whole policies file")
    args = parser.parse_args()

    input_dir = args.dir

    with open(f"{input_dir}/{args.policies_file}", "r", encoding="utf-8") as f:
        policies = json.load(f)

    with open(f"{input_dir}/scenario_params.json", "r", encoding="utf-8") as f:
        scenario_params = json.load(f)

    if args.seed_pool_size is not None:
        scenario_params['n_top_policies_for_cases'] = (
            args.seed_pool_size if args.seed_pool_size > 0 else None
        )

    kwargs = dict(input_dir=input_dir, policies=policies, scenario_params=scenario_params,
                  suffix=args.suffix)

    if args.action == "all":
        for name, fn in ACTIONS.items():
            print(f"\n--- Running {name} ---")
            fn(**kwargs, inference_settings=settings_for(name))
    else:
        ACTIONS[args.action](**kwargs, inference_settings=settings_for(args.action))


if __name__ == "__main__":
    main()
