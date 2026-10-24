"""
LLM Inference Package

A Python package for running LLM inference jobs with support for OpenAI, Gemini, and OpenRouter APIs.
Provides concurrent inference, batch processing, and result management capabilities.
"""

# Configuration
from .config import Config

# Inference (most common use)
from .inference_api import (
    query,
    run_inference,
)

# Repeated trials inference
from .repeated_trials import run_inference_repeated_trials

# Post-processing (converting raw outputs to probabilities)
from .postprocessing import (
    parse_custom_ids,
    parse_raw_results_file,
    logprobs_to_probabilities,
    results_to_probabilities,
    compute_binary_probabilities,
    postprocess,
)

# Local inference is imported on demand. It pulls in torch and transformers,
# which are an optional install (delta-ambiguity[local]) and weigh several
# gigabytes, so API-only use should not have to load them.
_LOCAL_EXPORTS = {
    "local_inference_logprob_tensors",
    "local_inference_binary_probabilities",
    "local_inference_simple",
}


def __getattr__(name):
    if name in _LOCAL_EXPORTS:
        try:
            from . import inference_local
        except ImportError as e:
            raise ImportError(
                f"{name} needs the optional 'local' extra: "
                "pip install 'delta-ambiguity[local] @ git+https://github.com/sally-dong/DELTA-ambiguity#subdirectory=delta'"
            ) from e
        return getattr(inference_local, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LOCAL_EXPORTS))

# Batch utilities (commonly used)
from .batch_utils import (
    get_batch,
    get_batch_output_dir,
    add_batch_args,
    print_batch_info,
    combine_jsonl_batches,
    combine_csv_batches,
    load_completed_ids_from_batches,
    filter_remaining_prompts,
)

