"""DEPRECATED import path, kept so existing Hydra configs resolve. Removed in Stage 12.

Serves only the config dotted paths `_target_: run_helpers.*`. Python code must import
noisyvis.experiments.hyperparams directly; nothing may import this module.
"""
from noisyvis.experiments.hyperparams import *  # noqa: F401,F403
