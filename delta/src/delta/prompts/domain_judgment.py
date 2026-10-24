def prompt_fn(*, rater:str, policy:str, case:str, scenario_params:dict, instruct_tuned:bool=False) -> list[str]:
    """
    Domain judgment template (6-prompt: 3 frames x 2 polarities).

    Required scenario_params keys:
        prompt_config: dict with keys:
            policy_label: e.g. "Policy", "Statute"
            case_label: e.g. "Student action", "Lawsuit", "Conversation"
            frames: list of 3 dicts, each with:
                intro: frame intro text
                positive_question: affirmative question
                negative_question: contradicting question
    """
    config = scenario_params["prompt_config"]
    policy_label = config["policy_label"]
    case_label = config["case_label"]

    prompts = []
    for frame in config["frames"]:
        base = (
            f"{rater}\n"
            f"{frame['intro']}\n"
            f"{policy_label}: {policy}\n"
            f"{case_label}: {case}\n\n"
            "After thinking carefully from their own point of view, they answer the following yes/no question.\n"
        )
        prompts.append(base + f"Question: {frame['positive_question']}")
        prompts.append(base + f"Question: {frame['negative_question']}")

    if instruct_tuned:
        return prompts[:2]

    prompts = [p + "\nAnswer:" for p in prompts]
    return prompts
