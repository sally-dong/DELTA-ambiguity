def prompt_fn(*, rater: str, policy: str, case: str, scenario_params: dict, instruct_tuned: bool = False) -> list[str]:
    """
    Adversarial Australian-republic template (aspiration/implementation variant):
    a stakeholder judges whether a proposed constitutional-reform implementation
    is consistent with a stated head-of-state aspiration for Australia.

    Required scenario_params keys:
        positive_question: the comply variant (yes = consistent with aspiration)
        negative_question: the violate variant (yes = inconsistent with aspiration)
    """

    base_context = (
        f"{rater}\n\n"
        f"They are evaluating whether a proposed constitutional-reform implementation is consistent with a stated aspiration for Australia's head-of-state arrangement.\n\n"
        f"CONSTITUTIONAL ASPIRATION:\n{policy}\n\n"
        f"PROPOSED IMPLEMENTATION:\n{case}\n\n"
        "After thinking carefully from their own point of view, they answer a yes/no question.\n"
    )

    prompt1 = base_context + f"Question: {scenario_params['positive_question']}"
    prompt2 = base_context + f"Question: {scenario_params['negative_question']}"

    if instruct_tuned:
        return [prompt1, prompt2]

    return [p + "\nAnswer:" for p in [prompt1, prompt2]]
