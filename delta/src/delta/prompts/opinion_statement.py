def prompt_fn(*, rater:str, policy:str, case:str, scenario_params:dict, instruct_tuned:bool=False) -> list[str]:
    """
    Opinion statement template.

    Required scenario_params keys:
        question: the policy question being reviewed
        positive_question: the affirmative question text
        negative_question: the contradicting question text
    """

    base_context = (
        f"{rater}\n\n"
        f"They are reviewing opinion statements on the question: \"{scenario_params['question']}\"\n\n"
        f"OPINION STATEMENT:\n{policy}\n\n"
        f"They consider how the opinion statement would translate into real-world policy actions. A candidate proposal is given below:\n\n"
        f"POLICY PROPOSAL:\n{case}\n\n"
        "After thinking carefully from their own point of view, they answer a yes/no question.\n"
    )

    prompt1 = base_context + f"Question: {scenario_params['positive_question']}"
    prompt2 = base_context + f"Question: {scenario_params['negative_question']}"

    if instruct_tuned:
        return [prompt1, prompt2]

    prompts = [prompt1, prompt2]
    prompts = [p + "\nAnswer:" for p in prompts]
    return prompts
