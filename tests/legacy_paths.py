"""The Stage 12 config dotted-path rewrite, as an exact comparison adapter (plan §8 Stage 12).

Stage 12 rewrites the config dotted paths from the temporary `src.*` / `run_helpers` compatibility
modules to the canonical `noisyvis.*` modules. The mapping below is exactly the plan's five rows. It is
used so that tests hold identically whichever spelling a config uses, and so that the frozen golden
`baselines/config_workflows.json`, which records the legacy spelling and is never re-recorded, can be
compared with outcomes resolved from canonical inputs.

This is a comparison adapter, not permission to rewrite historical expected values.

`canonicalise` rewrites a string only when it starts with one of the five full legacy module paths,
including the trailing `.`, so it maps only attributes of those exact modules. There is deliberately
no generic `src.` -> `noisyvis.` rule: any other string, including other `src.*` paths, is returned
unchanged.

Pure data and pure functions: nothing here imports the science packages.
"""

from __future__ import annotations

LEGACY_TO_CANONICAL = {
    "src.algorithms.Algorithms.": "noisyvis.algorithms.single_objective.",
    "src.algorithms.MOAlgorithms.": "noisyvis.algorithms.multi_objective.",
    "src.problems.ProblemScripts.": "noisyvis.problems.instances.",
    "src.problems.ViolationFunctions.": "noisyvis.problems.constraints.",
    "run_helpers.": "noisyvis.experiments.hyperparams.",
}


def canonicalise(value):
    """The canonical spelling of one dotted path; anything else is returned unchanged."""
    if not isinstance(value, str):
        return value
    for legacy, canonical in LEGACY_TO_CANONICAL.items():
        if value.startswith(legacy):
            return canonical + value[len(legacy):]
    return value


def canonicalise_tree(obj):
    """`canonicalise` applied to every string leaf of nested dicts, lists and tuples.

    Dict keys, container types and every non-string leaf are kept as they are.
    """
    if isinstance(obj, str):
        return canonicalise(obj)
    if isinstance(obj, dict):
        return {key: canonicalise_tree(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [canonicalise_tree(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(canonicalise_tree(item) for item in obj)
    return obj
