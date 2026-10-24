def prompt_fn(*, rater: str, policy: str, case: str, scenario_params: dict, instruct_tuned: bool = False) -> list[str]:
    """
    Campus speech template: a stakeholder judges whether a speech incident
    is permitted or prohibited under a given university speech policy.

    Required scenario_params keys:
        positive_question: the comply variant (yes = permitted)
        negative_question: the violate variant (yes = prohibited)
    """

    base_context = (
        f"{rater}\n\n"
        f"They are evaluating how a university speech policy applies to a specific incident on campus.\n\n"
        f"UNIVERSITY POLICY:\n{policy}\n\n"
        f"INCIDENT:\n{case}\n\n"
        "After thinking carefully from their own point of view, they answer a yes/no question.\n"
    )

    prompt1 = base_context + f"Question: {scenario_params['positive_question']}"
    prompt2 = base_context + f"Question: {scenario_params['negative_question']}"

    if instruct_tuned:
        return [prompt1, prompt2]

    return [p + "\nAnswer:" for p in [prompt1, prompt2]]
