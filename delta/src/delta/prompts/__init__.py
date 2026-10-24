import importlib.util
import os

from .opinion_statement import prompt_fn as opinion_statement_prompt_fn
from .domain_judgment import prompt_fn as domain_judgment_prompt_fn

TEMPLATES = {
    "opinion_statement": opinion_statement_prompt_fn,
    "domain_judgment": domain_judgment_prompt_fn,
}


def _load_from_dir(input_dir, fn_name='prompt_fn'):
    """Load a prompt function from custom_prompt_fn.py inside an experiment dir.

    Loaded by file path rather than module path, so the experiment directory
    does not need to be an importable package or sit under the working
    directory.
    """
    path = os.path.join(input_dir, 'custom_prompt_fn.py')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No prompt_template in scenario_params.json and no {path} to fall back on"
        )
    spec = importlib.util.spec_from_file_location('custom_prompt_fn', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


def get_prompt_fn(scenario_params, input_dir=None):
    """
    Get the prompt function for an experiment.

    If scenario_params contains a "prompt_template" key, use the named template.
    Otherwise, fall back to loading custom_prompt_fn.py from the data directory.
    """
    template_name = scenario_params.get("prompt_template")
    if template_name:
        return TEMPLATES[template_name]
    if input_dir is None:
        raise ValueError("No prompt_template in scenario_params and no input_dir for fallback")
    return _load_from_dir(input_dir)
