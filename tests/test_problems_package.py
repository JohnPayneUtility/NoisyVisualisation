"""Problems-package contracts, characterised before the Stage 9 split (plan Stage 9, Checkpoint 0).

Stage 9 renames `ViolationFunctions` -> `constraints` and `multiobjectiveFunctions` -> `knapsack_mo`, moves
`ProblemScripts` -> `instances` together with `instances_01_KP/` -> `instances/knapsack/` (anchored to
`INSTANCES_DIR`), and splits `FitnessFunctions` into `onemax`, `jump`, `knapsack` and `continuous`. The
reproducibility baselines execute only three evaluators and two knapsack instances, and the config gate
only checks that a name resolves. These tests pin what they cannot see:

 1. the 19 configured `fitness_fn` names resolve, through the runners' dynamic namespace, to unchanged
    definitions;
 2. the 31 top-level problem definitions exist exactly once (`mean_weight` exactly twice), unchanged;
 3. every global each of the 24 evaluators reads resolves to the same helper, module or shared object;
 4. every evaluator produces the same output, log records and RNG consumption on fixed inputs;
 5. every configured problem path resolves to its canonical module's object, whichever spelling
    (`src.problems.*` or `noisyvis.problems.*`) the config uses; until Stage 12 Checkpoint C deletes
    them, a separate temporary probe also pins the two `src.problems.*` forwarders' namespace and identity;
 6. the knapsack loader output for every instance, plus the stats/correlation helpers;
 7. the `knap_violation` behavioural divergence (D7) between the module-level and nested copies;
 8. the two `mean_weight` copies: equivalent, but distinct definitions;
 9. the problem imports of `Dashboard.py` and of `add_lon_nodes` (no config reaches four of them); the
    latter is read from viz/graph/lon.py, its post-Stage-10 location, with the pins unchanged across
    the graph-population split;
10. the byte content of the knapsack instance tree;
11. each definition lives in its pre-Stage-9 module or its intended Stage 9 module.

Everything is location-agnostic (the instance tree may be at either path, modules are found through the
objects runtime code uses), so the same tests hold before, during and after Stage 9. Checkpoint D
tightens the location contract to the post-split modules only.

The EXPECTED values are frozen literals captured once from a `git archive` of the untouched pre-Stage-9
commit PRE_STAGE_9_COMMIT, by running this file's probe against the archive in a fresh subprocess under
the runner's Python 3.11. They must never be re-derived from the implementation under test to make a
test pass. AST hashes are `sha256(ast.dump(<FunctionDef>))` as in test_core_library.py; `ast.dump`
depends on the Python version, so a runner upgrade legitimately needs them re-captured from that commit.

Any future PRE/post differential run must execute the archived and the current implementation in
separate fresh Python subprocesses, never two copies of `noisyvis` in one interpreter (sys.modules
contamination).

Imports of the science packages happen in one subprocess, so the pytest process never imports them.
The subprocess only reports what it observes; every expectation lives in this file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root
from legacy_paths import LEGACY_TO_CANONICAL

PRE_STAGE_9_COMMIT = "8424f5e5014d65f56ec942d51e158dbfe95e656c"

# ---------------------------------------------------------------- frozen at PRE_STAGE_9_COMMIT

# The top-level definitions of each pre-Stage-9 module, in source order.
PRE_STAGE_9_DEFINITIONS = {
    "FitnessFunctions.py": (
        "mean_weight", "bitflip_prior_noise",
        "OneMax_fitness", "OneMax_prior_bitflip_fitness", "OneMax_prior_mult_bitflip_fitness",
        "OneMax_prior_pq_bitwise_fitness", "OneMax_prior_1q_bitwise_fitness",
        "jump_fitness",
        "eval_ind_kp", "eval_noisy_kp_v1_simple", "eval_noisy_kp_v2_simple", "eval_noisy_kp_v1",
        "eval_noisy_kp_v1_penalty", "eval_noisy_kp_v2", "eval_noisy_kp_v2_penalty", "eval_noisy_kp_v3",
        "eval_noisy_kp_prior_bitflip", "eval_noisy_kp_prior_mult_bitflip", "eval_noisy_kp_pq_prior_bitwise",
        "eval_noisy_kp_1q_prior_bitwise",
        "rastrigin_eval", "birastrigin_eval", "ackley",
    ),
    "multiobjectiveFunctions.py": (
        "mean_weight", "eval_noisy_kp_v1_mo", "eval_noisy_kp_v1_mo_violation", "countingOnesCountingZeros",
    ),
    "ViolationFunctions.py": ("knap_violation",),
    "ProblemScripts.py": ("load_problem_KP", "get_knapsack_problem_stats", "interpret_correlation"),
}

# Every fitness_fn value in configs/ and tests/configs (derived from the corpus, not from this list).
CONFIGURED_FITNESS_FNS = frozenset({
    "OneMax_fitness", "OneMax_prior_bitflip_fitness", "OneMax_prior_mult_bitflip_fitness",
    "OneMax_prior_pq_bitwise_fitness", "OneMax_prior_1q_bitwise_fitness",
    "jump_fitness",
    "eval_ind_kp", "eval_noisy_kp_v1", "eval_noisy_kp_v1_penalty", "eval_noisy_kp_v2",
    "eval_noisy_kp_v2_penalty", "eval_noisy_kp_v3", "eval_noisy_kp_prior_bitflip",
    "eval_noisy_kp_1q_prior_bitwise",
    "rastrigin_eval", "birastrigin_eval",
    "eval_noisy_kp_v1_mo", "eval_noisy_kp_v1_mo_violation", "countingOnesCountingZeros",
})

# The complete evaluator API: the 19 configured names plus five no config uses.
UNCONFIGURED_EVALUATORS = (
    "eval_noisy_kp_v1_simple", "eval_noisy_kp_v2_simple", "eval_noisy_kp_prior_mult_bitflip",
    "eval_noisy_kp_pq_prior_bitwise", "ackley",
)
EVALUATORS = tuple(
    name for module in ("FitnessFunctions.py", "multiobjectiveFunctions.py")
    for name in PRE_STAGE_9_DEFINITIONS[module]
    if name not in ("mean_weight", "bitflip_prior_noise")
)

# sha256(ast.dump(FunctionDef)) of all 31 top-level definitions (both mean_weight copies share one hash).
DEFINITION_AST = {'OneMax_fitness': 'efe119c88de1ff4eae044e5b000fe4f55b78d5a789f756846d4ec2a6b11463cd',
 'OneMax_prior_1q_bitwise_fitness': '945e13dd71f63ca4475d8b8a90b686487ac495afd8067c8053382cda4afa5429',
 'OneMax_prior_bitflip_fitness': '0b59f1f4d4777eb688d96dc6d0d13f5a675f2dcf51805412c2e118b77e749145',
 'OneMax_prior_mult_bitflip_fitness': '805a680347ef10c0e97862cb4e83abd80dd96b728cb0eda755b554200c985ade',
 'OneMax_prior_pq_bitwise_fitness': '2e9cdaa23ca952ae5eb72c360f0a3866e118877aea558ef7c715efa5c937eb33',
 'ackley': '9b5a0e88c3e600459d6293fcfccc93553523a67a8acf02079e3a1d0afb2ed090',
 'birastrigin_eval': '23465fdf15c7cc14dd2fd9ac7ed8d543acdde7c98d16a7fe8df0fcc319685032',
 'bitflip_prior_noise': '79472046f93ddc94f337649c9fba89d68db8c77414976d6733214266a75f1853',
 'countingOnesCountingZeros': '4cd26cb55c4bc4152ae603bc733aa24ca92eda4928792d46b8077d5f38c6236e',
 'eval_ind_kp': '8f2531a6d23b80de72dc50e60ef4a7a94df4904b75610dd311c896592f5cb1f1',
 'eval_noisy_kp_1q_prior_bitwise': 'e7fab456eda1f6470baa37053d5d0c8483562673a0152b10c6c8a79e46d6dec6',
 'eval_noisy_kp_pq_prior_bitwise': '10fbabf30a804b709b3752bd0525b405c128e9ca1a1a918a2b0a86cfcac8ca29',
 'eval_noisy_kp_prior_bitflip': 'a8a587c99717bb2ad6e98d3ae4789033abaa0c60bab94016b70edac759baa07b',
 'eval_noisy_kp_prior_mult_bitflip': '4bbd661b666c4ddf69d79b2cc74abdaf93e89931e64423f76a7f265d8c10fa42',
 'eval_noisy_kp_v1': '0db44537754ad3d83418593188aea87bb59b734f9e56182d304ac12c78a7c650',
 'eval_noisy_kp_v1_mo': 'f184939911d835abca8b285ab0af81b83eb910b98f77d47c317c87b3d3498795',
 'eval_noisy_kp_v1_mo_violation': 'f147a08317ccc5f543ce2fb4f94f5208036b269da5f0e109cb4dbad894482e14',
 'eval_noisy_kp_v1_penalty': '81028aa2516d3504fdabdeae16c863d1ad24d1566b26a63fd0524230016e99eb',
 'eval_noisy_kp_v1_simple': '66c979ff6354a0d1dc03d3ac4b7acbbdf8d93220ff4b4e52408de29a6ec8f25d',
 'eval_noisy_kp_v2': 'ed7ee301c6c97f2094a073cb3ea2cb091d19405f86bf8abe31cbf838b3a13bc6',
 'eval_noisy_kp_v2_penalty': 'bd57e510f19034cc39ceadf47b914096dc76365fb048f5ac3695944cb5e42121',
 'eval_noisy_kp_v2_simple': 'fc0143c862042df73affdcfb2c812bea9b8b248783706582f34ab5c47f9dabdc',
 'eval_noisy_kp_v3': '595abc38d3a6ec39b69f981fb816cbb3221272433ff9695cab95c35b0e1fbd4a',
 'get_knapsack_problem_stats': 'a0db5f5afd50fad2e5f04be8d331eb3d52682974f6c1ccb506e78b4b3ce1412c',
 'interpret_correlation': '6d2c93e1d2d2614c477e13010f7ee8e2e62efec0ce802b14b0acfbbfd4ba7c0d',
 'jump_fitness': '01817406b81b90f5750882c16f98dc83c4c4f2edc40fde0f33150e3761d2d22b',
 'knap_violation': 'c2b4dd2b1f6173e70830553feed9c12c6329385c30c5f7f6e0e21ef9b672099c',
 'load_problem_KP': 'f68996637bc6062152cbad6dd5478afb74a75903c47306ceea150391ff4d3d6d',
 'mean_weight': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
 'rastrigin_eval': 'f1af9fee394ce5093b2804925538cc56c45d35f6d47c4ec1fbef98a2cdad7c6c'}

# The 24 evaluators, as getattr(noisyvis.problems, name) reaches them.
EVALUATOR_AST = {'OneMax_fitness': 'efe119c88de1ff4eae044e5b000fe4f55b78d5a789f756846d4ec2a6b11463cd',
 'OneMax_prior_bitflip_fitness': '0b59f1f4d4777eb688d96dc6d0d13f5a675f2dcf51805412c2e118b77e749145',
 'OneMax_prior_mult_bitflip_fitness': '805a680347ef10c0e97862cb4e83abd80dd96b728cb0eda755b554200c985ade',
 'OneMax_prior_pq_bitwise_fitness': '2e9cdaa23ca952ae5eb72c360f0a3866e118877aea558ef7c715efa5c937eb33',
 'OneMax_prior_1q_bitwise_fitness': '945e13dd71f63ca4475d8b8a90b686487ac495afd8067c8053382cda4afa5429',
 'jump_fitness': '01817406b81b90f5750882c16f98dc83c4c4f2edc40fde0f33150e3761d2d22b',
 'eval_ind_kp': '8f2531a6d23b80de72dc50e60ef4a7a94df4904b75610dd311c896592f5cb1f1',
 'eval_noisy_kp_v1_simple': '66c979ff6354a0d1dc03d3ac4b7acbbdf8d93220ff4b4e52408de29a6ec8f25d',
 'eval_noisy_kp_v2_simple': 'fc0143c862042df73affdcfb2c812bea9b8b248783706582f34ab5c47f9dabdc',
 'eval_noisy_kp_v1': '0db44537754ad3d83418593188aea87bb59b734f9e56182d304ac12c78a7c650',
 'eval_noisy_kp_v1_penalty': '81028aa2516d3504fdabdeae16c863d1ad24d1566b26a63fd0524230016e99eb',
 'eval_noisy_kp_v2': 'ed7ee301c6c97f2094a073cb3ea2cb091d19405f86bf8abe31cbf838b3a13bc6',
 'eval_noisy_kp_v2_penalty': 'bd57e510f19034cc39ceadf47b914096dc76365fb048f5ac3695944cb5e42121',
 'eval_noisy_kp_v3': '595abc38d3a6ec39b69f981fb816cbb3221272433ff9695cab95c35b0e1fbd4a',
 'eval_noisy_kp_prior_bitflip': 'a8a587c99717bb2ad6e98d3ae4789033abaa0c60bab94016b70edac759baa07b',
 'eval_noisy_kp_prior_mult_bitflip': '4bbd661b666c4ddf69d79b2cc74abdaf93e89931e64423f76a7f265d8c10fa42',
 'eval_noisy_kp_pq_prior_bitwise': '10fbabf30a804b709b3752bd0525b405c128e9ca1a1a918a2b0a86cfcac8ca29',
 'eval_noisy_kp_1q_prior_bitwise': 'e7fab456eda1f6470baa37053d5d0c8483562673a0152b10c6c8a79e46d6dec6',
 'rastrigin_eval': 'f1af9fee394ce5093b2804925538cc56c45d35f6d47c4ec1fbef98a2cdad7c6c',
 'birastrigin_eval': '23465fdf15c7cc14dd2fd9ac7ed8d543acdde7c98d16a7fe8df0fcc319685032',
 'ackley': '9b5a0e88c3e600459d6293fcfccc93553523a67a8acf02079e3a1d0afb2ed090',
 'eval_noisy_kp_v1_mo': 'f184939911d835abca8b285ab0af81b83eb910b98f77d47c317c87b3d3498795',
 'eval_noisy_kp_v1_mo_violation': 'f147a08317ccc5f543ce2fb4f94f5208036b269da5f0e109cb4dbad894482e14',
 'countingOnesCountingZeros': '4cd26cb55c4bc4152ae603bc733aa24ca92eda4928792d46b8077d5f38c6236e'}

# [repr(__defaults__), repr(__kwdefaults__)] per evaluator.
EVALUATOR_DEFAULTS = {'OneMax_fitness': ['(0,)', 'None'],
 'OneMax_prior_bitflip_fitness': ['(0,)', 'None'],
 'OneMax_prior_mult_bitflip_fitness': ['(0,)', 'None'],
 'OneMax_prior_pq_bitwise_fitness': ['(1, 1)', 'None'],
 'OneMax_prior_1q_bitwise_fitness': ['(1,)', 'None'],
 'jump_fitness': ['None', 'None'],
 'eval_ind_kp': ['(1,)', 'None'],
 'eval_noisy_kp_v1_simple': ['(0, 1)', 'None'],
 'eval_noisy_kp_v2_simple': ['(0, 1)', 'None'],
 'eval_noisy_kp_v1': ['(0, 1)', 'None'],
 'eval_noisy_kp_v1_penalty': ['(0, 10)', 'None'],
 'eval_noisy_kp_v2': ['(0, 1)', 'None'],
 'eval_noisy_kp_v2_penalty': ['(0, 10)', 'None'],
 'eval_noisy_kp_v3': ['(0, 1)', 'None'],
 'eval_noisy_kp_prior_bitflip': ['(0, 1)', 'None'],
 'eval_noisy_kp_prior_mult_bitflip': ['(0, 1)', 'None'],
 'eval_noisy_kp_pq_prior_bitwise': ['(0, 1, 1)', 'None'],
 'eval_noisy_kp_1q_prior_bitwise': ['(0, 10)', 'None'],
 'rastrigin_eval': ['(10, 0)', 'None'],
 'birastrigin_eval': ['(1, None, 0)', 'None'],
 'ackley': ['(20, 0.2, 6.283185307179586, 0)', 'None'],
 'eval_noisy_kp_v1_mo': ['(0, 0, 0)', 'None'],
 'eval_noisy_kp_v1_mo_violation': ['(0, 0, 0)', 'None'],
 'countingOnesCountingZeros': ['(0, 0)', 'None']}

# Every global each evaluator reads (def statement, body and nested functions) and what it resolves to.
EVALUATOR_GLOBALS = {'OneMax_fitness': {'get_active_logger': {'kind': 'function',
                                          'name': 'get_active_logger',
                                          'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                          'is_noisyvis_algorithms_export': False,
                                          'is_logger_holder_function': True},
                    'random': {'kind': 'module', 'module': 'random'},
                    'sum': {'kind': 'builtin'}},
 'OneMax_prior_bitflip_fitness': {'bitflip_prior_noise': {'kind': 'function',
                                                          'name': 'bitflip_prior_noise',
                                                          'ast': '79472046f93ddc94f337649c9fba89d68db8c77414976d6733214266a75f1853',
                                                          'is_noisyvis_algorithms_export': False,
                                                          'is_logger_holder_function': False},
                                  'get_active_logger': {'kind': 'function',
                                                        'name': 'get_active_logger',
                                                        'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                        'is_noisyvis_algorithms_export': False,
                                                        'is_logger_holder_function': True},
                                  'len': {'kind': 'builtin'},
                                  'list': {'kind': 'builtin'},
                                  'sum': {'kind': 'builtin'}},
 'OneMax_prior_mult_bitflip_fitness': {'get_active_logger': {'kind': 'function',
                                                             'name': 'get_active_logger',
                                                             'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                             'is_noisyvis_algorithms_export': False,
                                                             'is_logger_holder_function': True},
                                       'list': {'kind': 'builtin'},
                                       'random_bit_flip': {'kind': 'function',
                                                           'name': 'random_bit_flip',
                                                           'ast': '9b0d90c2dce16fd6c4d322178bc6e179dea2727f737c06ecfee3bad240bd203a',
                                                           'is_noisyvis_algorithms_export': True,
                                                           'is_logger_holder_function': False},
                                       'sum': {'kind': 'builtin'}},
 'OneMax_prior_pq_bitwise_fitness': {'get_active_logger': {'kind': 'function',
                                                           'name': 'get_active_logger',
                                                           'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                           'is_noisyvis_algorithms_export': False,
                                                           'is_logger_holder_function': True},
                                     'len': {'kind': 'builtin'},
                                     'list': {'kind': 'builtin'},
                                     'random': {'kind': 'module', 'module': 'random'},
                                     'sum': {'kind': 'builtin'}},
 'OneMax_prior_1q_bitwise_fitness': {'get_active_logger': {'kind': 'function',
                                                           'name': 'get_active_logger',
                                                           'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                           'is_noisyvis_algorithms_export': False,
                                                           'is_logger_holder_function': True},
                                     'len': {'kind': 'builtin'},
                                     'random': {'kind': 'module', 'module': 'random'},
                                     'sum': {'kind': 'builtin'}},
 'jump_fitness': {'len': {'kind': 'builtin'}, 'sum': {'kind': 'builtin'}},
 'eval_ind_kp': {'len': {'kind': 'builtin'}, 'range': {'kind': 'builtin'}, 'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v1_simple': {'len': {'kind': 'builtin'},
                             'random': {'kind': 'module', 'module': 'random'},
                             'range': {'kind': 'builtin'},
                             'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v2_simple': {'len': {'kind': 'builtin'},
                             'random': {'kind': 'module', 'module': 'random'},
                             'range': {'kind': 'builtin'},
                             'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v1': {'get_active_logger': {'kind': 'function',
                                            'name': 'get_active_logger',
                                            'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                            'is_noisyvis_algorithms_export': False,
                                            'is_logger_holder_function': True},
                      'len': {'kind': 'builtin'},
                      'mean_weight': {'kind': 'function',
                                      'name': 'mean_weight',
                                      'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                      'is_noisyvis_algorithms_export': False,
                                      'is_logger_holder_function': False},
                      'random': {'kind': 'module', 'module': 'random'},
                      'range': {'kind': 'builtin'},
                      'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v1_penalty': {'get_active_logger': {'kind': 'function',
                                                    'name': 'get_active_logger',
                                                    'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                    'is_noisyvis_algorithms_export': False,
                                                    'is_logger_holder_function': True},
                              'len': {'kind': 'builtin'},
                              'mean_weight': {'kind': 'function',
                                              'name': 'mean_weight',
                                              'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                              'is_noisyvis_algorithms_export': False,
                                              'is_logger_holder_function': False},
                              'random': {'kind': 'module', 'module': 'random'},
                              'range': {'kind': 'builtin'},
                              'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v2': {'get_active_logger': {'kind': 'function',
                                            'name': 'get_active_logger',
                                            'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                            'is_noisyvis_algorithms_export': False,
                                            'is_logger_holder_function': True},
                      'len': {'kind': 'builtin'},
                      'mean_weight': {'kind': 'function',
                                      'name': 'mean_weight',
                                      'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                      'is_noisyvis_algorithms_export': False,
                                      'is_logger_holder_function': False},
                      'random': {'kind': 'module', 'module': 'random'},
                      'range': {'kind': 'builtin'},
                      'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v2_penalty': {'get_active_logger': {'kind': 'function',
                                                    'name': 'get_active_logger',
                                                    'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                    'is_noisyvis_algorithms_export': False,
                                                    'is_logger_holder_function': True},
                              'len': {'kind': 'builtin'},
                              'mean_weight': {'kind': 'function',
                                              'name': 'mean_weight',
                                              'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                              'is_noisyvis_algorithms_export': False,
                                              'is_logger_holder_function': False},
                              'random': {'kind': 'module', 'module': 'random'},
                              'range': {'kind': 'builtin'},
                              'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v3': {'get_active_logger': {'kind': 'function',
                                            'name': 'get_active_logger',
                                            'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                            'is_noisyvis_algorithms_export': False,
                                            'is_logger_holder_function': True},
                      'len': {'kind': 'builtin'},
                      'mean_weight': {'kind': 'function',
                                      'name': 'mean_weight',
                                      'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                      'is_noisyvis_algorithms_export': False,
                                      'is_logger_holder_function': False},
                      'random': {'kind': 'module', 'module': 'random'},
                      'range': {'kind': 'builtin'},
                      'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_prior_bitflip': {'bitflip_prior_noise': {'kind': 'function',
                                                         'name': 'bitflip_prior_noise',
                                                         'ast': '79472046f93ddc94f337649c9fba89d68db8c77414976d6733214266a75f1853',
                                                         'is_noisyvis_algorithms_export': False,
                                                         'is_logger_holder_function': False},
                                 'get_active_logger': {'kind': 'function',
                                                       'name': 'get_active_logger',
                                                       'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                       'is_noisyvis_algorithms_export': False,
                                                       'is_logger_holder_function': True},
                                 'len': {'kind': 'builtin'},
                                 'list': {'kind': 'builtin'},
                                 'range': {'kind': 'builtin'},
                                 'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_prior_mult_bitflip': {'get_active_logger': {'kind': 'function',
                                                            'name': 'get_active_logger',
                                                            'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                            'is_noisyvis_algorithms_export': False,
                                                            'is_logger_holder_function': True},
                                      'len': {'kind': 'builtin'},
                                      'list': {'kind': 'builtin'},
                                      'random_bit_flip': {'kind': 'function',
                                                          'name': 'random_bit_flip',
                                                          'ast': '9b0d90c2dce16fd6c4d322178bc6e179dea2727f737c06ecfee3bad240bd203a',
                                                          'is_noisyvis_algorithms_export': True,
                                                          'is_logger_holder_function': False},
                                      'range': {'kind': 'builtin'},
                                      'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_pq_prior_bitwise': {'get_active_logger': {'kind': 'function',
                                                          'name': 'get_active_logger',
                                                          'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                          'is_noisyvis_algorithms_export': False,
                                                          'is_logger_holder_function': True},
                                    'len': {'kind': 'builtin'},
                                    'list': {'kind': 'builtin'},
                                    'random': {'kind': 'module', 'module': 'random'},
                                    'range': {'kind': 'builtin'},
                                    'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_1q_prior_bitwise': {'get_active_logger': {'kind': 'function',
                                                          'name': 'get_active_logger',
                                                          'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                                          'is_noisyvis_algorithms_export': False,
                                                          'is_logger_holder_function': True},
                                    'len': {'kind': 'builtin'},
                                    'random': {'kind': 'module', 'module': 'random'},
                                    'range': {'kind': 'builtin'},
                                    'sum': {'kind': 'builtin'}},
 'rastrigin_eval': {'get_active_logger': {'kind': 'function',
                                          'name': 'get_active_logger',
                                          'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                          'is_noisyvis_algorithms_export': False,
                                          'is_logger_holder_function': True},
                    'len': {'kind': 'builtin'},
                    'np': {'kind': 'module', 'module': 'numpy'},
                    'random': {'kind': 'module', 'module': 'random'},
                    'sum': {'kind': 'builtin'}},
 'birastrigin_eval': {'get_active_logger': {'kind': 'function',
                                            'name': 'get_active_logger',
                                            'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                            'is_noisyvis_algorithms_export': False,
                                            'is_logger_holder_function': True},
                      'len': {'kind': 'builtin'},
                      'min': {'kind': 'builtin'},
                      'np': {'kind': 'module', 'module': 'numpy'},
                      'random': {'kind': 'module', 'module': 'random'},
                      'sum': {'kind': 'builtin'}},
 'ackley': {'get_active_logger': {'kind': 'function',
                                  'name': 'get_active_logger',
                                  'ast': '0a4b6daad9583b09df7f3bca02177feaf4fdd21b8b00fa59d06cc734de3fc99f',
                                  'is_noisyvis_algorithms_export': False,
                                  'is_logger_holder_function': True},
            'len': {'kind': 'builtin'},
            'np': {'kind': 'module', 'module': 'numpy'},
            'random': {'kind': 'module', 'module': 'random'}},
 'eval_noisy_kp_v1_mo': {'len': {'kind': 'builtin'},
                         'mean_weight': {'kind': 'function',
                                         'name': 'mean_weight',
                                         'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                         'is_noisyvis_algorithms_export': False,
                                         'is_logger_holder_function': False},
                         'random': {'kind': 'module', 'module': 'random'},
                         'range': {'kind': 'builtin'},
                         'sum': {'kind': 'builtin'}},
 'eval_noisy_kp_v1_mo_violation': {'float': {'kind': 'builtin'},
                                   'int': {'kind': 'builtin'},
                                   'len': {'kind': 'builtin'},
                                   'max': {'kind': 'builtin'},
                                   'mean_weight': {'kind': 'function',
                                                   'name': 'mean_weight',
                                                   'ast': '817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca',
                                                   'is_noisyvis_algorithms_export': False,
                                                   'is_logger_holder_function': False},
                                   'random': {'kind': 'module', 'module': 'random'},
                                   'range': {'kind': 'builtin'},
                                   'sum': {'kind': 'builtin'}},
 'countingOnesCountingZeros': {'len': {'kind': 'builtin'},
                               'random': {'kind': 'module', 'module': 'random'},
                               'sum': {'kind': 'builtin'}}}

# sha256 of each isolated case record: output, error, log records, post-call random/numpy RNG state, input and kwargs unchanged, logger cleared.
EVALUATOR_OUTPUTS = {'00:OneMax_fitness[bits20]seed=20260917': '2b809de5d50215fdee34ad4b6e53ba98e2a9921540f93ffd42e5b33a383239ae',
 '00:OneMax_fitness[bits20]seed=20260917+logger': '4557c7e2f68526b65e720422b5dc8060e2d4b28d67ed97b44b2a688b363c084b',
 '00:OneMax_fitness[bits20]seed=1': '54cbcf2b9955e92220cfe6da6f7a123e62a8ad0ef069b43f1e598c48f76ce068',
 '00:OneMax_fitness[bits20]seed=1+logger': '1b42858fc498a45033d1f4d73d260e85bd510204d0a9bdb5955b23c98dc1de67',
 '00:OneMax_fitness[bits20]seed=2': 'b1d160b8c6bf8f18ac2577ee008991618af0165aa454b1b9f314ad2e6ce183c0',
 '00:OneMax_fitness[bits20]seed=2+logger': '3a6db0e98b35456dfd8d0c22b69d78427941f31f2e130059d222f0c6ae8185f6',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=20260917': '42958f8a3cc8839e4df94d823c4f5d6f4a71f4d04757b4698812c063fd89c7d4',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=20260917+logger': 'd7b37291813b1243f706f4d3aa08751d8f5d0233d6a85b47ee82ff51628d515e',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=1': 'ab59a60e2d6c732dc8eac99a69ddb8ff4afe163fe874c0622f0c6e2df5263a18',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=1+logger': '685db50d56f8f0ee198e68af664ccaedfc04db60b1d02e1ba29c2e2aeee60f38',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=2': '29a74fe881b45dc98b3b08846a818aa8e9f59e265eba1f0f1cb04dbe305abe3f',
 '01:OneMax_prior_bitflip_fitness[bits20]seed=2+logger': 'ac15415d59a23d8f697e62947c88d4615386b93a60c264791beeb47fb7aca678',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=20260917': '21d591c983a642f72faa7d8a0cfd7086d981fc5ecd938d01c061f69c790a5197',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=20260917+logger': 'a29aff7246ed4e106ef6b1afe978ebc231c376067dfee84cf88372acde847633',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=1': 'ab59a60e2d6c732dc8eac99a69ddb8ff4afe163fe874c0622f0c6e2df5263a18',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=1+logger': '8285d2b32b36fb8ed5183a66d41c31abc77ba613a24ea5165eb050581584717d',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=2': 'e0efd5b65aca9120024d7e31c96f5f574fbc5ab406d706f488a63be49f54c3bb',
 '02:OneMax_prior_mult_bitflip_fitness[bits20]seed=2+logger': '770c38276a913ceab216801d195fc563f6c0189ef7ac5f19ebf2db161cbf010c',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=20260917': '42958f8a3cc8839e4df94d823c4f5d6f4a71f4d04757b4698812c063fd89c7d4',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=20260917+logger': 'd7b37291813b1243f706f4d3aa08751d8f5d0233d6a85b47ee82ff51628d515e',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=1': '220b29131251bf7038f51390a74801e485507338140b494fb3140f0d78ace035',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=1+logger': '8d108158422a8af95cdcb50a6c3787cfc75914f1685966ac54ff110e4f92f621',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=2': '29a74fe881b45dc98b3b08846a818aa8e9f59e265eba1f0f1cb04dbe305abe3f',
 '03:OneMax_prior_pq_bitwise_fitness[bits20]seed=2+logger': 'ac15415d59a23d8f697e62947c88d4615386b93a60c264791beeb47fb7aca678',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=20260917': '42958f8a3cc8839e4df94d823c4f5d6f4a71f4d04757b4698812c063fd89c7d4',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=20260917+logger': 'd7b37291813b1243f706f4d3aa08751d8f5d0233d6a85b47ee82ff51628d515e',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=1': 'e9e2111eb4eacd53e88b74ac27c22eaaece66f64c18b3f0c21fcc6f4394e052f',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=1+logger': '3577777909712dd47c2257fa50e6df695439ce91b74078fee57badd4c42865dd',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=2': '29a74fe881b45dc98b3b08846a818aa8e9f59e265eba1f0f1cb04dbe305abe3f',
 '04:OneMax_prior_pq_bitwise_fitness[bits20]seed=2+logger': 'ac15415d59a23d8f697e62947c88d4615386b93a60c264791beeb47fb7aca678',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=20260917': 'a0a8cfe62b8862fb0895c498626fd906e59c6fbcb554399e27eec53804db14dc',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=20260917+logger': '37616c2c18fb1128ec3a886ca3220582325101e3ba1912b9114fe0db997914df',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=1': '66be570b19b7b871d1d3dc50a0ca1e97a0ba0eebafa3c6121c77ad6088579fa6',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=1+logger': '9528b929e3700845196d2b988b9aa4c94414161e1dbd2713141520c729c765e8',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=2': '5d9d6ef22b2e4cd8a91d9ad9d1ad3495c14a9008b7d598a2251530b66c8b7479',
 '05:OneMax_prior_1q_bitwise_fitness[bits20]seed=2+logger': '0cc9427dd883fc8511fcf86071ff0dd95a673d38715208f20cc035d2351275da',
 '06:jump_fitness[bits20]seed=20260917': 'f790f6e4bd4772a5c645d7a045fa9fc945042de7b730ba100a1d2bff31da214c',
 '06:jump_fitness[bits20]seed=20260917+logger': '348eead145ea9fcab716cc0f031d71a92ba89b5e224bfedd28689b003aa4ba24',
 '06:jump_fitness[bits20]seed=1': '226edf38bdd76a0484e080b40a630023b60413be3c2d7c003d354d0edb29e4ad',
 '06:jump_fitness[bits20]seed=1+logger': '4e7e570a0f0a44e181fba2d9440379dae4e2415a0c4b475805bee4addf13e925',
 '06:jump_fitness[bits20]seed=2': '40457e9811cdaa32ac0ff569179fb6b70445f8be65a92fb6539f591f68496b78',
 '06:jump_fitness[bits20]seed=2+logger': '42cc769ab0d1f2d5811d0ed84aaeb1ef91feeea5b8250ec7a9f5436073fc4a07',
 '06:jump_fitness[ones20]seed=20260917': 'fc63111255a0001a905a2c726476b82ff814317d70957a24b9c2c36ae9d480fa',
 '06:jump_fitness[ones20]seed=20260917+logger': '68da988bb11ae3bc56c874a5008d33bef071b06bcac03544766beb96b7958b4d',
 '06:jump_fitness[ones20]seed=1': 'bbd2f4cbd214e683ec19446cd18894f6f189be11211a924d1a0053316b244458',
 '06:jump_fitness[ones20]seed=1+logger': 'a7ef79a747990f34cf09ddf821fdc179ff7f08d1a677e132d4895a128f9e0943',
 '06:jump_fitness[ones20]seed=2': '626bb9a147b63978971e368b2562cab86ecb3f67daf76634b7b4010485c9c2be',
 '06:jump_fitness[ones20]seed=2+logger': 'bbe2e6c9c17ec3629ef289b60d1ec6c9eca05d660b2aeff06c20be223cc8aa2d',
 '06:jump_fitness[ones19]seed=20260917': '5be0bd8ed0e1901818b916422f26714a8092c75d25b19156df07f31a522b2d1d',
 '06:jump_fitness[ones19]seed=20260917+logger': 'a8c7ce5f56c7ab965f8de72a39ceb771e4a1180c422daf16248b7687691d2ea1',
 '06:jump_fitness[ones19]seed=1': 'd98c12c2b14ec3e565a6655f10370ccd7caa7a005f58dc44e1e0c3279e97c338',
 '06:jump_fitness[ones19]seed=1+logger': 'c9331cec570f20aeaa33beefb79cc6c632fb57e01f278793e9917b71cf3e9aa9',
 '06:jump_fitness[ones19]seed=2': '2c7f45b53fffc062b796f886d192e3964ee327dabb92f8928e664dfa61dbcac5',
 '06:jump_fitness[ones19]seed=2+logger': '7a253429855ff57989f983e1bebcec499e0bcaf559a0f54b7f6b6ab40c85093a',
 '07:eval_ind_kp[kp_feasible]seed=20260917': '1a421bbd59ab36569e5dd4b14a3c095ad3f45a72efc5d08890067caaacf4cec0',
 '07:eval_ind_kp[kp_feasible]seed=20260917+logger': 'd41ebb35e33b6e223379b48cc9a0fbaf6c0fe45d5636f93db6e0dfeef5b7fbc9',
 '07:eval_ind_kp[kp_feasible]seed=1': '0be37e67f40eb8912294eedf911aaf4e763d89cccfa0879c26500ce584fc4db8',
 '07:eval_ind_kp[kp_feasible]seed=1+logger': 'd92a1c6d5c3f0e4414bd7efdab0e5fb9227eb430741e531d3b0bb68a08ada76a',
 '07:eval_ind_kp[kp_feasible]seed=2': 'd66490158f25305a5339c5e6fcb6a3fbd7d024d04841eb40017c796d17a9ac61',
 '07:eval_ind_kp[kp_feasible]seed=2+logger': '01f1be116837f85807d8291b38b9d8a663edb4e419e2e58705def88baed33f59',
 '07:eval_ind_kp[kp_infeasible]seed=20260917': 'a2b8d4b6eb1491fb006f7fe7a7191577bfe723d0cbc5862e9791cff97de417de',
 '07:eval_ind_kp[kp_infeasible]seed=20260917+logger': 'b192d6a88f7d88316a271ce3bbbe8b683c45969b406d7a2adfb7533f8965e98a',
 '07:eval_ind_kp[kp_infeasible]seed=1': '2933d4f939b9f7bd97e01681d31bb1078ff0cc2a9185918eaec42b7806606d14',
 '07:eval_ind_kp[kp_infeasible]seed=1+logger': '31d13f8832c484718a52440a896100957c2f0b980e09e5d6b62e16bf4d3d5829',
 '07:eval_ind_kp[kp_infeasible]seed=2': '0aa368890205a75ae43274751466a61dcffb0d9170c5260cc2226d731304db06',
 '07:eval_ind_kp[kp_infeasible]seed=2+logger': '9d27b0f858747805d46376bd228995030013e6090d5b7ba17218f860db163841',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=20260917': '284f3618dd547fd45bf670578adf8b0c4d900e35c44a0395fd5b3d933b6bbc75',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=20260917+logger': '11f53b84d6e37c91596d1eb4aa00e7596db50d64f5e075896336f5ba7ba16d47',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=1': 'dd0c43d527acdec7d4ca799cda84192a45ff9015833fd292a7558530d1b73138',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=1+logger': '89ece141644885cb615eb559bc50b43b2e8cb85f318874dd0a26f457bde21708',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=2': '0bcc084fca8483684598d3922957fbc25e8b6618ff4635ab9f58cf972f0a0150',
 '08:eval_noisy_kp_v1_simple[kp_feasible]seed=2+logger': '9350e0cd6ddc25cffee29008d0c5001ed6b3c858f03174a3a0d2ee6198c3afe6',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=20260917': '66cb507a44177e1d98a425660d51024dc590b726bd9a67e5543c58986d722bc7',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=20260917+logger': '35ff3eb9ee1e41c505961e6c940ccc72dca45b56327cd4320e0042776faa90ea',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=1': '32a236658865ffb5bec8936ee144aab4800030742c1a419f5cac61bffb9b7fde',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=1+logger': '6bec849e66b7f03b29580807ef20c70c56de2fdaf19c346c5bb6f8a9e836b3a2',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=2': '985e510304bb8241bdb466d0224295e5d712079f4f66cbfd2f8a7b2f3c0de94c',
 '08:eval_noisy_kp_v1_simple[kp_infeasible]seed=2+logger': '1893784894988e39985503941e1539a44a2c46dd3c657741b7c516bb4cda83af',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=20260917': '284f3618dd547fd45bf670578adf8b0c4d900e35c44a0395fd5b3d933b6bbc75',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=20260917+logger': '11f53b84d6e37c91596d1eb4aa00e7596db50d64f5e075896336f5ba7ba16d47',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=1': '5fbab2fe91bb9573c3ee03778b8ebefda7d4b3e319bdd6d9e8a518517f70c396',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=1+logger': '10f0f933247ea964691fa49ffeeb4e8611b2b54940d6c7e3f2a92efe88240287',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=2': '76d3df82e5d7f3bbdf8dfb164e9e14267495f9f707ca6250db7f697c6c403af9',
 '09:eval_noisy_kp_v2_simple[kp_feasible]seed=2+logger': '6863d66fff3cb3f1e7c498437def5a4f1d2479e69a2a99378a84bb464c37c48e',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=20260917': '66cb507a44177e1d98a425660d51024dc590b726bd9a67e5543c58986d722bc7',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=20260917+logger': '35ff3eb9ee1e41c505961e6c940ccc72dca45b56327cd4320e0042776faa90ea',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=1': '32a236658865ffb5bec8936ee144aab4800030742c1a419f5cac61bffb9b7fde',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=1+logger': '6bec849e66b7f03b29580807ef20c70c56de2fdaf19c346c5bb6f8a9e836b3a2',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=2': '985e510304bb8241bdb466d0224295e5d712079f4f66cbfd2f8a7b2f3c0de94c',
 '09:eval_noisy_kp_v2_simple[kp_infeasible]seed=2+logger': '1893784894988e39985503941e1539a44a2c46dd3c657741b7c516bb4cda83af',
 '10:eval_noisy_kp_v1[kp_feasible]seed=20260917': '5aed156a2311b78811e02eb75ed2ecd7c8997a6cba3ecd642253d31c4ed8b9f6',
 '10:eval_noisy_kp_v1[kp_feasible]seed=20260917+logger': '827a6e870c55af5accfad98aab12fcd143a02feb905d2f9d909cee25ba6eb998',
 '10:eval_noisy_kp_v1[kp_feasible]seed=1': '6a90a680b49d425ca0d8c74648bc5f4a16ccc223840a8ad62f9e12a8dfea7beb',
 '10:eval_noisy_kp_v1[kp_feasible]seed=1+logger': '82074b7ac25dd069eaab4636d9ad039271facd85793db76af2ea00c176bcada1',
 '10:eval_noisy_kp_v1[kp_feasible]seed=2': '275b78d41e90dc99c29c2953930de1462e4023e56b39b80552a92b880b2fd814',
 '10:eval_noisy_kp_v1[kp_feasible]seed=2+logger': 'f829d5841a89cdf0aada8ca3d00e93a54d9c4b643643d043c3e4ddc2ac7aaccb',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=20260917': '66cb507a44177e1d98a425660d51024dc590b726bd9a67e5543c58986d722bc7',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=20260917+logger': '890a651fdf834382c2a88519795da2cc5eeffa4d9c9cf32dc6b07def58bb4795',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=1': '32a236658865ffb5bec8936ee144aab4800030742c1a419f5cac61bffb9b7fde',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=1+logger': '1ce3efe7e5190af047ba07dbcadd586fbe188500bc30b63df1190fd3e0b34efe',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=2': '985e510304bb8241bdb466d0224295e5d712079f4f66cbfd2f8a7b2f3c0de94c',
 '10:eval_noisy_kp_v1[kp_infeasible]seed=2+logger': '24f18f26e1e5fda07a2999ba299e153e9cb369b43e67b4ed4f98639106f591cd',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=20260917': '5aed156a2311b78811e02eb75ed2ecd7c8997a6cba3ecd642253d31c4ed8b9f6',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=20260917+logger': '827a6e870c55af5accfad98aab12fcd143a02feb905d2f9d909cee25ba6eb998',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=1': '6a90a680b49d425ca0d8c74648bc5f4a16ccc223840a8ad62f9e12a8dfea7beb',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=1+logger': '82074b7ac25dd069eaab4636d9ad039271facd85793db76af2ea00c176bcada1',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=2': '275b78d41e90dc99c29c2953930de1462e4023e56b39b80552a92b880b2fd814',
 '11:eval_noisy_kp_v1_penalty[kp_feasible]seed=2+logger': 'f829d5841a89cdf0aada8ca3d00e93a54d9c4b643643d043c3e4ddc2ac7aaccb',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=20260917': '5e0562646c1d2fbee800c638199465c1a4742ee5be677079b029d403f401d6ec',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=20260917+logger': 'cd22fd95cf0e31e28656a0198ef3d4f13b603b57781f8d660a715f7607435901',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=1': '255e53bf18fb4391c9a651ea64792f18cb2381f388d89fc76c371df5602bc934',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=1+logger': 'c49d3d4649bb20a7aebceacdcbd4c37893379dd05f940e04999576dfc51bf2e6',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=2': '4f622cdf4e9c5e6129f12da131e36f3405b518976108aa3baaf193f1435b89d7',
 '11:eval_noisy_kp_v1_penalty[kp_infeasible]seed=2+logger': 'e89753c8a7bf4dc47aaad3f54a58f2f715fa88a1098634d82dbc4d724d6f1a94',
 '12:eval_noisy_kp_v2[kp_feasible]seed=20260917': '5aed156a2311b78811e02eb75ed2ecd7c8997a6cba3ecd642253d31c4ed8b9f6',
 '12:eval_noisy_kp_v2[kp_feasible]seed=20260917+logger': '827a6e870c55af5accfad98aab12fcd143a02feb905d2f9d909cee25ba6eb998',
 '12:eval_noisy_kp_v2[kp_feasible]seed=1': '6a90a680b49d425ca0d8c74648bc5f4a16ccc223840a8ad62f9e12a8dfea7beb',
 '12:eval_noisy_kp_v2[kp_feasible]seed=1+logger': '82074b7ac25dd069eaab4636d9ad039271facd85793db76af2ea00c176bcada1',
 '12:eval_noisy_kp_v2[kp_feasible]seed=2': '76d3df82e5d7f3bbdf8dfb164e9e14267495f9f707ca6250db7f697c6c403af9',
 '12:eval_noisy_kp_v2[kp_feasible]seed=2+logger': '157470e36fff89387e777f11311b9da27311e174a6e2fcb51dec1de59018aa64',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=20260917': '66cb507a44177e1d98a425660d51024dc590b726bd9a67e5543c58986d722bc7',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=20260917+logger': '890a651fdf834382c2a88519795da2cc5eeffa4d9c9cf32dc6b07def58bb4795',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=1': '32a236658865ffb5bec8936ee144aab4800030742c1a419f5cac61bffb9b7fde',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=1+logger': '1ce3efe7e5190af047ba07dbcadd586fbe188500bc30b63df1190fd3e0b34efe',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=2': '985e510304bb8241bdb466d0224295e5d712079f4f66cbfd2f8a7b2f3c0de94c',
 '12:eval_noisy_kp_v2[kp_infeasible]seed=2+logger': '24f18f26e1e5fda07a2999ba299e153e9cb369b43e67b4ed4f98639106f591cd',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=20260917': '81d8a60802e97f23db14e378369127dda504024d30393f0c226ec0e053180ffb',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=20260917+logger': '1bf63a73bfa55a447269b190f755909e2e7225609b8162357a5932dd6ae725e1',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=1': '4b761d8943aa99b906f34e2d72130f660f60d97ccd3d4391bf231ffc4bd07927',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=1+logger': '258ebef849fe60b89776227e77f31c6133b4df81425d50fab41e1d0d478d9fde',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=2': '65d6116d588a0896dfedf88b3c2b72dc23c52ded0bc41a71ce2112fb29b6b08a',
 '13:eval_noisy_kp_v2_penalty[kp_feasible]seed=2+logger': '421d7d5faf429b436dcfc9ee4fb2e46ea52bdb4a92e5e41cc92a5724525fd121',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=20260917': '473f2a7854aca3b2bc27ddbf94b3d96012713959fa7ebf35e5300afe811f4236',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=20260917+logger': 'dec60c89ee74cbbeffa431c8c1dd72b4269ed5197f6956bc9be87b9f5a856bc5',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=1': '296ab7b3bca974b5c2d4d95f775b86470f27ed4b63f656b1cfa8e4a06fdaa467',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=1+logger': '181e73736884e9f160505de4d634c9d0be88c93ae236aa8b678f503b323a336e',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=2': '673aa7daf61d5b9b2d10a84b7e16f8b0960d9c99c45285274d7a9c108d5622c5',
 '13:eval_noisy_kp_v2_penalty[kp_infeasible]seed=2+logger': 'b78677387cddc5299ad96a72c7000fc94f3c58dec4edef41eae8f48652eb4602',
 '14:eval_noisy_kp_v3[kp_feasible]seed=20260917': '33766d7e53ca96ae98be10c6dcb46ec033d96399de66a8796cfc2f5ad3b72b5a',
 '14:eval_noisy_kp_v3[kp_feasible]seed=20260917+logger': '2f00031c4b054b3e92d259c47454aeddde9f8f9cb68d3d7a5239dc2b35de609a',
 '14:eval_noisy_kp_v3[kp_feasible]seed=1': 'f9ada035b9bed5bacd48f8ac4c41c9e1b870d0387290ed8f2056d78414bb9f41',
 '14:eval_noisy_kp_v3[kp_feasible]seed=1+logger': '02b16ed476a7d89ed89fd06c7260640544bfedf0535ad1b3d5cc8fd81f01d471',
 '14:eval_noisy_kp_v3[kp_feasible]seed=2': 'bdb91b0e7d94337a383f2ea601a57bd529579378c3abcf30c8846aefd650b087',
 '14:eval_noisy_kp_v3[kp_feasible]seed=2+logger': 'de117a7d1502a7db4f1060608e7674be283c7b0fac8d92d200fc46b7e662b4a3',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=20260917': '20a3bfcb02eaef1e23bd8ae71bc154e2ef0dd686d2ee4c30adb45c01673142dc',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=20260917+logger': '8da64413aebf264b8ceeba3e1ce450939a2cb0e2cc2f15a3efbdcaaf8fc62611',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=1': '70502484bdd595e931e0f6c434ed9734bec9483b6adbe4b0c1d35d06d24fc2f8',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=1+logger': '9af8999c62c59e899a6d26eff659b3a0e386c653e6b1dc3631e2aeff6b558de5',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=2': '1074810f42001f812fe1b18a99b93d184f48fdda3c975f9ea0d3d7ded209de09',
 '14:eval_noisy_kp_v3[kp_infeasible]seed=2+logger': '94059f558a80b96ac6f53a8968737e89f2532589e870d7f5cbaf8555ef4bb599',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=20260917': '45ae61c956e1b49e7a712a08beffa4a109dbc4daeaafa9c71b113fdc4b49970f',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=20260917+logger': 'b40e21c3d48859c87c2fa628af8f10b3bebad5970fb0c76fbdced6f303d4ea00',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=1': '113f33c7fef43489e893c8199cb4157ed1330cc86193a13e36433b49f82960e3',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=1+logger': 'e26c6c33ba8fb3c17df5efc4c4a176338ea193c9b9e19846154354816bf4b54b',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=2': 'e1c1792648409d75181ac43050191500b990fd5da3842bff73a5b457eca240e3',
 '15:eval_noisy_kp_prior_bitflip[kp_feasible]seed=2+logger': '244492486a05856854f27f11de292e6a8c2391a7fad1870de77ce1b6bfb6032d',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=20260917': '300f88735ede7e4f89ee0753dd5d052300b184048f009854f0752da1e10b0147',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=20260917+logger': '8a18374f0bc7a7dc1a2397115ad9d33044fd3f74e010050ff357f76cb306c0b4',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=1': 'e345a30966c0cac10640101d7883f2a7d757d35285f33eb12d9e04967f1ab38f',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=1+logger': 'a8af78d989b7eabf1be1c4261506b143b45c231d0f66cda70790fb40de3cb09b',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=2': '9b727efcfcbda8e24527397642817ed54e40b1a7a7e1c29b46577820f46b5c89',
 '15:eval_noisy_kp_prior_bitflip[kp_infeasible]seed=2+logger': 'fd18f0386d3557b173a9661d2a074d23606975658d10e36963251391d0c6645d',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=20260917': 'c209d313c180afd75695f9ee4464ea86beea5779589aa7489bb73935b8749a14',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=20260917+logger': '9e2b2c200de757dbc95476d542dffe938acac5f5decef148d7cae1fa5dc1dc3a',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=1': 'd282f617ec5e008c10a60e4f1edc1f8e15ff2cea8f861b095f94f122bad98c28',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=1+logger': 'a00546bcd1e4dd8d78c903f3906ade7f86637679e22887621496f14fc491462c',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=2': '568e4d14ba1dd4bdc91e3a5ca841677e391e53e6dc0379a64e9bd2201cff0fd4',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_feasible]seed=2+logger': '6e664e116f4bdb33ff8ab52f0d5b31d0b990087e7229ff4f18294cb12b533174',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=20260917': '6619710046c6402c7afa29597dc004bfeff2987f66918cef691022c98d4e69b4',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=20260917+logger': '7891f8d605c36a0ddc0365c4d63121cd35efc744a7d4ddca37853b059006df98',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=1': 'e3756010db91f5495b22194f07ca7c5f4d40bbcd56c9b41279cb68676b48d10c',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=1+logger': '7fbf8b31afe86ac26a33b2468e9cd0ba071ca6af3d1297ba36d73a2157f5ccad',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=2': 'c28a8f2c40a892826a2900000d1bf3c1d0d1931067567387373bfbb3da30b0ce',
 '16:eval_noisy_kp_prior_mult_bitflip[kp_infeasible]seed=2+logger': 'f289f916f1318ce4b1fc58eb2a9285844f04b2813bb13c97ffba48782af0bfbb',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=20260917': '45ae61c956e1b49e7a712a08beffa4a109dbc4daeaafa9c71b113fdc4b49970f',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=20260917+logger': 'b40e21c3d48859c87c2fa628af8f10b3bebad5970fb0c76fbdced6f303d4ea00',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=1': '8a2b824eca7c53689c636782b730c3c79924025613b6ed551af88f77268e8b3e',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=1+logger': 'a581409e16fbb68c982fcd8d2c3e8887068f023080a8cb6c4bd19f1ede62b68f',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=2': 'e1c1792648409d75181ac43050191500b990fd5da3842bff73a5b457eca240e3',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_feasible]seed=2+logger': '244492486a05856854f27f11de292e6a8c2391a7fad1870de77ce1b6bfb6032d',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=20260917': '300f88735ede7e4f89ee0753dd5d052300b184048f009854f0752da1e10b0147',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=20260917+logger': '8a18374f0bc7a7dc1a2397115ad9d33044fd3f74e010050ff357f76cb306c0b4',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=1': '78053910557c2f7d14d3cd6bef30a7a13a311e3d1fc384572fb7dd1930893ac7',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=1+logger': 'd5a7fd0b91ff5506e180c8bf75baaee8dd0ea340c6ef9475a1a08216ee1f8ef3',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=2': '9b727efcfcbda8e24527397642817ed54e40b1a7a7e1c29b46577820f46b5c89',
 '17:eval_noisy_kp_pq_prior_bitwise[kp_infeasible]seed=2+logger': 'fd18f0386d3557b173a9661d2a074d23606975658d10e36963251391d0c6645d',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=20260917': '4ddf6a9481bfedf1d274e717398fc32cd46f0587b6b678c060a83c78ffa1b24b',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=20260917+logger': 'a34f1a93920ee1b0663c5e485f82a890ae1e12fd2050cac62ae88bea6fdf465c',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=1': 'fb59bc49f02271c39d9084090e450f6112c40df0c7b6a47739c9cb7062e653a6',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=1+logger': '45d65f872fafb7b0491eeac3cc8fd60659a44f56044df21afa517fa817905203',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=2': '94e86fd6212566c22f1df58f5b9ec7b8667f98fa65cadf2a883aa653a692b7c7',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_feasible]seed=2+logger': '4f04fcfcf293127a60993bc182e1a7858959eacc581e874a7255d9e676f92d6d',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=20260917': '43ce9c8881cea9e6ff9e4a3d7dd47eee582f226bfa4e585e93839cfc007c8b18',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=20260917+logger': '2f548cc4af8530f2ed1ae3111d1027d7fc591c78ae04f644b1826465c7119675',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=1': 'c290fbf9141b1cff2dea16492a04e99e59897b7af26a2c744b309a4ca71cdfe6',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=1+logger': '4a94aad4e1aa175fb4a71cb19567a020b9b8808c1d9b6feee3544131974f914b',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=2': '3a8fa666ce6eda723ab4bdffbc839d0cb9a552bed12ae1381970fede1927122b',
 '18:eval_noisy_kp_1q_prior_bitwise[kp_infeasible]seed=2+logger': 'bf9ad2fec921719de464297f6d4873d1dc9d68228d25a2721276389ed36e5b6d',
 '19:rastrigin_eval[vec2]seed=20260917': '3a83bda5138a451d34cceb08671ff41272ad5c31f880fbfe0546e6a930fc7db6',
 '19:rastrigin_eval[vec2]seed=20260917+logger': '7bc0f4d834dd4b4f4d45e52804534cab190e935f30e9a7d7cdaaad6b485ef815',
 '19:rastrigin_eval[vec2]seed=1': '9d26d6f4c8c925a291b8f0eb24df033ded35f39996c6d7524a986fec059ef50a',
 '19:rastrigin_eval[vec2]seed=1+logger': 'cd22249daa05563965a31256266fb6ce9982463b4d61be3985eca9fbf567c87e',
 '19:rastrigin_eval[vec2]seed=2': '7c875eb8ed26c27bfb6b00dfdc8562ea0bc0648031bcf9a23db9d49368a630bc',
 '19:rastrigin_eval[vec2]seed=2+logger': '8c7c11082aeeaaf55e010b086125a106ed539944ee0ca83d8a8c3c0566f7c6e6',
 '20:birastrigin_eval[vec2]seed=20260917': 'f1e215b51c295ee8a31b9ca838aa7699c62998ffd457252fe0741e2eedabe8df',
 '20:birastrigin_eval[vec2]seed=20260917+logger': '87196ef614c92065585c3ba8fbbd3bf106ffd1efe886fca569df38cb6572740e',
 '20:birastrigin_eval[vec2]seed=1': 'c5b13649876b2bafb1abd27f806ecc83e3832c0d643083ca9de8923b76820353',
 '20:birastrigin_eval[vec2]seed=1+logger': '0b14503d60c4099f0ea0a284dda0e8e8ad8dd561d2fe4df55cb2396c3f1d32ef',
 '20:birastrigin_eval[vec2]seed=2': '8de240ade6bb049b3320acb81d69ff151e75552d00b86e6b4e89754daba5a013',
 '20:birastrigin_eval[vec2]seed=2+logger': '11a003b06b3792315b6d17359139cd3a3a386135aab6cf41ca7bb851b7310767',
 '21:ackley[vec2]seed=20260917': '61f668f8008273eff606bdbc247c71b576e5f3006073677b70f1a44ecede18ad',
 '21:ackley[vec2]seed=20260917+logger': 'ca1064a6a78b4a5eb67f58d08b634e0f5009e9600c7b46cfac5cf33787775b76',
 '21:ackley[vec2]seed=1': '11e0196b478f1242817b4665b68302aad7904e9adcff4f839ac5cfd7d92538d0',
 '21:ackley[vec2]seed=1+logger': 'b7e12897d6feca899293d8dee840e318571c8e41aa25b6bc337a79fda6bf147c',
 '21:ackley[vec2]seed=2': 'b0421f93a590add6dcebc55f0c07066070f6e6ea166b8d2ae05d645da70e0d1e',
 '21:ackley[vec2]seed=2+logger': '8ca3ea800ebc19702754ecc9150a0b6a86dfd14c43c9094dd21cd6b2ea876ee5',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=20260917': '3928d04cc81efb3e5a1bfa4e883cee496fbcadff1b089a38d01dc4e84f08c14d',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=20260917+logger': '5273aa986067caa58e9af0e1d96cee3300b16d11c8815b76e5cbef784a00e3d3',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=1': '59fbada25f7464580356f389a4bcef3d25c31dc0d0acc6b863f4c11222aac852',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=1+logger': '80641d207a70911efeac549eea06c1d0562bd092446db43cfb2daf14c6668aa3',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=2': 'd2bdb5b0d28662bc97a68614769e60fc49b84de847849088ec50379848f32fc7',
 '22:eval_noisy_kp_v1_mo[kp_feasible]seed=2+logger': '0dcba3fad0eccf30f6dd2b3596295858207ec04e7c586551c0a262d035d59bab',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=20260917': '9d46588ab3d6079f416c94474dc3af273b5b58cd5882549fe8b3f3f8d9a9a9ca',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=20260917+logger': '5f30509b140db1e3a933b8033588976074d0933ab202221067b974925f75991f',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=1': 'cb6146a835c287bf066a23b74c24e06e719de85cbde67b93b07ee3db6fb8f318',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=1+logger': '51bcbb851a9bcbd32733b92abec37179dc8399f055d9f4d2b9e0215cf7bab39e',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=2': '685579e224b4a47229d649a73116e372213b5bebf111aca2535874445f2367af',
 '22:eval_noisy_kp_v1_mo[kp_infeasible]seed=2+logger': 'e73ac24b3b5824737999c6c6c5aff03a193de8e6aed44ff404d416c2107bbbad',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=20260917': 'ee5974e01030757a782656714c1a7fcabc76e4dab2a6751b7566974d29648aea',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=20260917+logger': '00642cacff43ad9e4e453bb29d79d07273328e955b205ea4fa79f9ae03866fe3',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=1': '674b4612592a097e88a860f284055e630db4a57fd11aa8f407d73ab853a67dd4',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=1+logger': 'a20737ac84d8ed8cef61d57476218d2e4c393ed6b1bbecb38ef31cfcca7be02d',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=2': 'd7e958d7f04373b10d7baf694147d695be41e6354c8104efa6941a1c724bc8cc',
 '23:eval_noisy_kp_v1_mo[kp_infeasible]seed=2+logger': '5af76b33a31207041db175a5e06acf7cfe1d036fb991f0f1e14231073801de7e',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=20260917': '5a74e99bbe11ac34456842c4b44508e9b7e2619baf7174dff842160bf9364255',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=20260917+logger': '56564ab2a683bfa9aaea5e7de4498cd3555300b14b4cdb56433847627ee40e05',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=1': '7da020ec49d9880367aa6cad9d8621afc8a318318913a54e087e18e815d10374',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=1+logger': 'd9ebb3315dd793c2e47fa15025bae6127093317210fc49902b365712fecbf7d8',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=2': 'fbfe0713b1cc84e22a191f724c7ce732a331899e102326fc1eb8b588a9feb67c',
 '24:eval_noisy_kp_v1_mo_violation[kp_feasible]seed=2+logger': 'a2f78551bbf34ec63bcfa64b372bcfcfc55d071f974c1c052cd30a8694c4629b',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=20260917': '4708894286b2ef6c84e73acbf17b291a21d0133fa59304cc463773c665203b4d',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=20260917+logger': '05f5f501410968f45bc4dc1ffd7d4270fef0af6458f72c2415cb14be46aadb84',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=1': '81080955aa86c21074ee19f22627709fade41cdbe907fbf6b933508736661be8',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=1+logger': '8603e0a87c4f4a23aaef8988ebd10999f072af45332f8d963b35d7d400cb0900',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=2': '55cf9b7ce4443c41573d8d3aaf8e0e95c682191aa7ed9eb4c5f7516ed7ba5408',
 '24:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=2+logger': 'a6f83acf7af6f6b08d98bc21831862bea34e6697f010e8fa2777cf0179d8d72e',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=20260917': 'b5ac2d87e74874cb04aa0a126a63c1aa3aae489ea27b48d54abcb1f06323e6c5',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=20260917+logger': 'e01ba40b01799a2782050423101ba0a42c874af53fb065bd50228c5a5e83719c',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=1': 'b6b1a54e54cbfd90b6f03fa4395471050c27abf2e61dedffb93f04e25cfb5a7b',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=1+logger': '34f2269bac1cc3486797c01aa248b559d347d8043c446cdeb8e52b75678ebb86',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=2': 'b39421db5c22bab7580f6c68d76c28690e32cb2495899e2974770b6a2e2fb8d7',
 '25:eval_noisy_kp_v1_mo_violation[kp_infeasible]seed=2+logger': 'bf5f3260ace6d373896e36cfb9b967fc0b9c459013b803be0901224763743f35',
 '26:countingOnesCountingZeros[bits20]seed=20260917': 'b6a07b93ac3ece5c669609feaf31587b6b6e7cf9e0d6266cf9080784978ab348',
 '26:countingOnesCountingZeros[bits20]seed=20260917+logger': '765e3ef77bde9f00b1e12487a4383a8629b78ed465957fe9ef0d30d123de25e9',
 '26:countingOnesCountingZeros[bits20]seed=1': '41659ab49ed1f04824df7e21a0471430d9d09405dd34a85868cc7d28dde1bb7a',
 '26:countingOnesCountingZeros[bits20]seed=1+logger': '0c2e697bb6bbbee7299a87123cb667b47e3f5ab1f9aaffce346df3fd77dacf71',
 '26:countingOnesCountingZeros[bits20]seed=2': '996a81b4e939d2f0f6cbc10299cf2a8a85ceafc8ce88284cc371074991d4dda6',
 '26:countingOnesCountingZeros[bits20]seed=2+logger': '32594d6209c2ecf98d4cc653a5a1d9c9841195e3258f41d95816181accc92361'}

# Public namespace of each src.problems forwarder (== its canonical module's).
FORWARDER_PUBLIC_NAMES = {'ProblemScripts': frozenset({'get_knapsack_problem_stats', 'load_problem_KP', 'interpret_correlation'}),
 'ViolationFunctions': frozenset({'knap_violation'})}

# load_problem_KP(pid) through src.problems.ProblemScripts, type-preserving digest (or the raised error).
LOADER_DIGESTS = {'f10_l-d_kp_20_879': {'digest': '652a2101f226d6326b1e02c5734ac4f89a5ba0af8bedbf6d9fa96019cdfedc97'},
 'f1_l-d_kp_10_269': {'digest': 'fcc6c6ffb379a96d23945fdbcea913177c526acca679e611c1258a27cafff058'},
 'f2_l-d_kp_20_878': {'digest': '115313ebf26925e21e365408af5d16f2e5886077200277c7c5f28bb80fa93602'},
 'f3_l-d_kp_4_20': {'digest': 'a15252fdf2e1beee4a6bf3feb200cd30c2d432cf55a5e29b21e5cc637c32a219'},
 'f4_l-d_kp_4_11': {'digest': 'e586bbb504164285475d25302ab5c523967bd693b561f5e642f382932e508087'},
 'f5_l-d_kp_15_375': {'error': "ValueError: could not convert string '0.125126' to int64 at row 1, column "
                               '1.'},
 'f6_l-d_kp_10_60': {'digest': 'd33bde60611acb870b9aeacb4c93b74ce09690e7d1bae5b7268f239ff169ef30'},
 'f7_l-d_kp_7_50': {'digest': '69fca5878bb97c3cec57034fe71c2dbe6d140f3a453f683f42eb6afe40129bcc'},
 'f8_l-d_kp_23_10000': {'digest': 'a1eeadc29cf1532dfeb0a2f813f1bc55a6e0b6ae7aa6e3b6bb7dea3a23f448db'},
 'f9_l-d_kp_5_80': {'digest': '5629d927d2f406db472f3a419cef5b829dda9e56d224133ef3c269ac4b5a8f10'},
 'knapPI_1_10000_1000_1': {'digest': 'f18ff5e4b50dca8de2752b66be1680f660e589072c2367d689af5c70f553ee50'},
 'knapPI_1_1000_1000_1': {'digest': '33ed662c371c7eb682f681e6e5f67af58d035fdf78b4a36c6dd452036813bfc0'},
 'knapPI_1_100_1000_1': {'digest': 'c0b68a7c5b1eccdb2f3c3440bc8f58d9125c3654dbc292c563008171d73de0f0'},
 'knapPI_1_2000_1000_1': {'digest': '31cadda50bacf7924694023426f350b889f9f7df01ae722bbf68b2c865eb40a2'},
 'knapPI_1_200_1000_1': {'digest': '4ec4758b2f5dc08d1518db1bed639ee0f9c9cf63d79073cc4ddf8d7bd55e94e7'},
 'knapPI_1_5000_1000_1': {'digest': '1f4fee57b3993fdf2a766a9a0b92c0dd857ed44b02db1a22a2b968ba37994c9e'},
 'knapPI_1_500_1000_1': {'digest': 'a3d0cd27ce07e766b7a6dfa8181fdc716004c8e65ba3a45fc3032e48ca81e63a'},
 'knapPI_2_10000_1000_1': {'digest': '20f8d080808eb5d64bff0aaea5ff1d013c0babbc4e7faddb45ec4d05358a622b'},
 'knapPI_2_1000_1000_1': {'digest': '949be3c367c1b9a6388d9375b3098dcbde5b2ffebb14c6991dacae986131a852'},
 'knapPI_2_100_1000_1': {'digest': '729c374a94c4dc0ae3f4790a8e506503baa7e009a02f5a4a93e12ba309f3ada6'},
 'knapPI_2_2000_1000_1': {'digest': 'fa5c450f2cb7578cb50159a7bb42ceb43d522bc80ed0b5a11deeeac6fada6207'},
 'knapPI_2_200_1000_1': {'digest': '4878d79f7221f066b5428633a99438139cca88b9e7480383fa6e517ee7850f1e'},
 'knapPI_2_5000_1000_1': {'digest': 'bf75c6e1e278e63cb73d91f834d716baedffe00d714c86816c286e22caf5a434'},
 'knapPI_2_500_1000_1': {'digest': 'eacc8d3b6ba6d72ff97109a347a80357bcc9599023ca4c26bfd647191ebedf5e'},
 'knapPI_3_10000_1000_1': {'digest': '7a06ecaf5e2e39ba824abdfb236a979af32c27ccd258de3842c170ea53912c72'},
 'knapPI_3_1000_1000_1': {'digest': '995b0dd8c1ea9edabb21f041a1d2998ba77fed667b83a8bb35a14c5b596bc823'},
 'knapPI_3_100_1000_1': {'digest': 'ac28add6baa5c281dc9b0ccf024150d727124725ebae25c36157a1c495f2f735'},
 'knapPI_3_2000_1000_1': {'digest': '092b45ac5728e53afde9ab17365248cdf07e2449c18a0f201d001de519702ac4'},
 'knapPI_3_200_1000_1': {'digest': '4847fc0083c0dc46209f3ccf399945490dcccb8b11eff010af77a6281b5f5477'},
 'knapPI_3_5000_1000_1': {'digest': '5debbe947f52ede7c1e29384ffa5072c810fc4392ab4208182f967e64e5dc443'},
 'knapPI_3_500_1000_1': {'digest': '6dc6d8a324b6859c5f884652e9e3aa48a9b8dc4121a6afdd3be3e1ac7e8294ba'}}

# [n_items, capacity] per PID; None where loading raises.
LOADER_SHAPES = {'f10_l-d_kp_20_879': [20, 879],
 'f1_l-d_kp_10_269': [10, 269],
 'f2_l-d_kp_20_878': [20, 878],
 'f3_l-d_kp_4_20': [4, 20],
 'f4_l-d_kp_4_11': [4, 11],
 'f5_l-d_kp_15_375': None,
 'f6_l-d_kp_10_60': [10, 60],
 'f7_l-d_kp_7_50': [7, 50],
 'f8_l-d_kp_23_10000': [23, 10000],
 'f9_l-d_kp_5_80': [5, 80],
 'knapPI_1_10000_1000_1': [10000, 49877],
 'knapPI_1_1000_1000_1': [1000, 5002],
 'knapPI_1_100_1000_1': [100, 995],
 'knapPI_1_2000_1000_1': [2000, 10011],
 'knapPI_1_200_1000_1': [200, 1008],
 'knapPI_1_5000_1000_1': [5000, 25016],
 'knapPI_1_500_1000_1': [500, 2543],
 'knapPI_2_10000_1000_1': [10000, 49877],
 'knapPI_2_1000_1000_1': [1000, 5002],
 'knapPI_2_100_1000_1': [100, 995],
 'knapPI_2_2000_1000_1': [2000, 10011],
 'knapPI_2_200_1000_1': [200, 1008],
 'knapPI_2_5000_1000_1': [5000, 25016],
 'knapPI_2_500_1000_1': [500, 2543],
 'knapPI_3_10000_1000_1': [10000, 49519],
 'knapPI_3_1000_1000_1': [1000, 4990],
 'knapPI_3_100_1000_1': [100, 997],
 'knapPI_3_2000_1000_1': [2000, 9819],
 'knapPI_3_200_1000_1': [200, 997],
 'knapPI_3_5000_1000_1': [5000, 24805],
 'knapPI_3_500_1000_1': [500, 2517]}

# load_problem_KP on an unknown PID.
LOADER_MISSING_ERROR = 'FileNotFoundError: No knapsack instance found for PID: no_such_instance_pid'

# get_knapsack_problem_stats(pid) digests (or the raised error).
STATS_DIGESTS = {'f1_l-d_kp_10_269': {'digest': '09553a4cbeb172a728e16be463a14101c7afcb7b449e96713e03a99235b984f6'},
 'f5_l-d_kp_15_375': {'error': "ValueError: could not convert string '0.125126' to int64 at row 1, column "
                               '1.'},
 'f8_l-d_kp_23_10000': {'digest': '5993dd36237596777a2326f88b33e9c880a201c76f232be0728999db3407c515'},
 'knapPI_1_100_1000_1': {'digest': '0eddcd7dd983bb20beebad4f908c0724b3868089ead04cad85367f7eff356bf7'},
 'knapPI_3_10000_1000_1': {'digest': '6048184b06fe99208422d8de9e5f408572561b32b3ec3835a1a74d0e628f713f'}}

# interpret_correlation on (nan, 0.5, -0.5, 0.05, 0.1, -0.1).
CORRELATION_LABELS = ['n/a', 'positive', 'negative', 'neutral', 'positive', 'negative']

# knap_violation (config-reachable, signed) versus the copy nested in eval_noisy_kp_v1_mo_violation (clamped), on KP10: feasible weight 233, infeasible weight 348, capacity 269.
VIOLATION_PINS = {'module_level': {'kp_feasible': {'float': '-0x1.2000000000000p+5'},
                  'kp_infeasible': {'float': '0x1.3c00000000000p+6'}},
 'nested_in_mo_evaluator': {'kp_feasible': {'int': '0'}, 'kp_infeasible': {'float': '0x1.3c00000000000p+6'}},
 'module_return': 'return float(total_w - capacity)',
 'nested_return': ['return max(0, float(total_w - capacity))'],
 'module_ast': 'c2b4dd2b1f6173e70830553feed9c12c6329385c30c5f7f6e0e21ef9b672099c',
 'nested_ast': ['b302859f2dd1bee3d86ebacb6735a79b9fcb58791276ea42c2356b68e1dee724']}

# The two mean_weight copies as their seven users reach them.
MEAN_WEIGHT_PINS = {'single-objective': {'users': ['eval_noisy_kp_v1',
                                'eval_noisy_kp_v1_penalty',
                                'eval_noisy_kp_v2',
                                'eval_noisy_kp_v2_penalty',
                                'eval_noisy_kp_v3'],
                      'one_object': True,
                      'ast': ['817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca'],
                      'value_on_kp10': {'numpy.float64': '0x1.af33333333333p+5'}},
 'multi-objective': {'users': ['eval_noisy_kp_v1_mo', 'eval_noisy_kp_v1_mo_violation'],
                     'one_object': True,
                     'ast': ['817c46251ff23c87428d0b32c665feceab3722d6963f3e65b9167c98fffe40ca'],
                     'value_on_kp10': {'numpy.float64': '0x1.af33333333333p+5'}},
 'copies_are_distinct_objects': True}

# Problem imports of Dashboard.py (module level) and graph_builder.add_lon_nodes (lazy).
IMPORT_PINS = {'dashboard': {'explicit_is_package_object': {'get_knapsack_problem_stats': True,
                                              'interpret_correlation': True,
                                              'load_problem_KP': True},
               'star_names': ['OneMax_fitness',
                              'OneMax_prior_1q_bitwise_fitness',
                              'OneMax_prior_bitflip_fitness',
                              'OneMax_prior_mult_bitflip_fitness',
                              'OneMax_prior_pq_bitwise_fitness',
                              'ackley',
                              'birastrigin_eval',
                              'bitflip_prior_noise',
                              'eval_ind_kp',
                              'eval_noisy_kp_1q_prior_bitwise',
                              'eval_noisy_kp_pq_prior_bitwise',
                              'eval_noisy_kp_prior_bitflip',
                              'eval_noisy_kp_prior_mult_bitflip',
                              'eval_noisy_kp_v1',
                              'eval_noisy_kp_v1_penalty',
                              'eval_noisy_kp_v1_simple',
                              'eval_noisy_kp_v2',
                              'eval_noisy_kp_v2_penalty',
                              'eval_noisy_kp_v2_simple',
                              'eval_noisy_kp_v3',
                              'get_active_logger',
                              'jump_fitness',
                              'mean_weight',
                              'np',
                              'random',
                              'random_bit_flip',
                              'rastrigin_eval']},
 'graph_builder.add_lon_nodes': {'explicit_is_package_object': {'eval_noisy_kp_1q_prior_bitwise': True,
                                                                'eval_noisy_kp_pq_prior_bitwise': True,
                                                                'eval_noisy_kp_prior_bitflip': True,
                                                                'eval_noisy_kp_prior_mult_bitflip': True,
                                                                'eval_noisy_kp_v1': True,
                                                                'eval_noisy_kp_v1_penalty': True,
                                                                'eval_noisy_kp_v1_simple': True,
                                                                'eval_noisy_kp_v2': True,
                                                                'eval_noisy_kp_v2_penalty': True,
                                                                'eval_noisy_kp_v2_simple': True,
                                                                'eval_noisy_kp_v3': True,
                                                                'load_problem_KP': True},
                                 'star_names': []}}

# relative path -> [size, sha256] for every file of the instance tree.
INSTANCE_MANIFEST = {'Source Link.txt': [61, '363cff848e0b65fde982dbb7369b38fcab7401310f76c4244ae58c1f0fa4350a'],
 'large_scale-optimum/knapPI_1_10000_1000_1': [6,
                                               'ba31400f97fdbab82c5c2312a0d840764e2ebdf0695e0ec69299c63782063405'],
 'large_scale-optimum/knapPI_1_1000_1000_1': [5,
                                              '55b0cbd5c7e742cf94fc7f4e959d300e452e4a201dfe90601561d2907a42a26a'],
 'large_scale-optimum/knapPI_1_100_1000_1': [4,
                                             '89086fb5ed35208767a70e9369540f621abefc50c830fcef89e0a50c57584f8c'],
 'large_scale-optimum/knapPI_1_2000_1000_1': [6,
                                              '8f375995385035c25d012b1c5c160568efca4ad035d7056b8ca9ad69fe728b16'],
 'large_scale-optimum/knapPI_1_200_1000_1': [5,
                                             '9250fa378fe370ad3bb71585336505fa26915515f2886b10742dadc0c6b3de2d'],
 'large_scale-optimum/knapPI_1_5000_1000_1': [6,
                                              '4752c1e04419c72d83722d9319c97b6d0a264e54e1fa88d8da664f295d5bc245'],
 'large_scale-optimum/knapPI_1_500_1000_1': [5,
                                             '34f47dfa5f0323b60d836c1af00f720b0d73e28e0ae83fd0d9a03a94c68ba3c0'],
 'large_scale-optimum/knapPI_2_10000_1000_1': [5,
                                               '1b24916bb3e2739e212d6b089a666047920a81408b15d5e5f63ad53d76279de0'],
 'large_scale-optimum/knapPI_2_1000_1000_1': [4,
                                              '214538a798d46607ed8c5bb7cb54c13f9bc164789f576296189559feeef5b3ad'],
 'large_scale-optimum/knapPI_2_100_1000_1': [4,
                                             'c8f5600f3eb7d801a067e02c477719d37f560491c4dca56eacc11ca755eb1ac6'],
 'large_scale-optimum/knapPI_2_2000_1000_1': [5,
                                              'a2a91c8170dd733076b84470011029229133534a9705f1cb2435afa4bfbb0c4e'],
 'large_scale-optimum/knapPI_2_200_1000_1': [4,
                                             'ca05bc2bf73745fee2f3d493297d4f90930477b484d2c08d7dc899d9a574de89'],
 'large_scale-optimum/knapPI_2_5000_1000_1': [5,
                                              '4871ed65666c521a2c95e0040bd1a0028e7a24546eff77aed64440f5f8947d0c'],
 'large_scale-optimum/knapPI_2_500_1000_1': [4,
                                             '0454a8d72f0fda8244e7ec5754978037816c3118b0a1316d1cd352bd5d76c159'],
 'large_scale-optimum/knapPI_3_10000_1000_1': [6,
                                               'e5db9e037a8d8ceacb7736c9d8e9b96e90bcf82784ceb621f856d39f7dcae735'],
 'large_scale-optimum/knapPI_3_1000_1000_1': [5,
                                              '82d5a2a5add9cc511072270b271308f4877e21061ec007d68f73b32eccdd3d0e'],
 'large_scale-optimum/knapPI_3_100_1000_1': [4,
                                             'cbbe2e41fff1a2f04968bdaeedff3b78085afca2dc4623870bdc2dff3aac6747'],
 'large_scale-optimum/knapPI_3_2000_1000_1': [5,
                                              'fb6047fead22cd55b476e2eb5790ca343bfe0888b1bb7772893216f3e34d840d'],
 'large_scale-optimum/knapPI_3_200_1000_1': [4,
                                             '1498172a195a73ec4ac7550b66f0a5e639e9c19d6b51787a4b1a93aa374dea44'],
 'large_scale-optimum/knapPI_3_5000_1000_1': [5,
                                              'e420a98188b3b5d13668cab6936f9030333a6a40301ca1d8dac0eda19b08165e'],
 'large_scale-optimum/knapPI_3_500_1000_1': [4,
                                             'cc830be9dae15b28f32c242fad7ed204df8d18c761555a3b710cf319d5ad4cb8'],
 'large_scale/knapPI_1_10000_1000_1': [97931,
                                       '4f8a22eea4f26c7a6310a2ce403eb8527c2661d1e27f98247ec71d5b3705307c'],
 'large_scale/knapPI_1_1000_1000_1': [9803,
                                      '382bce422e5f5aaa11305b5ed920fd430eb08750dc23883cdcf93c0746face43'],
 'large_scale/knapPI_1_100_1000_1': [986, '6662297e409393da38ea83dc9951edec372f6f27dd6625aa1c63991f6f127e77'],
 'large_scale/knapPI_1_2000_1000_1': [19599,
                                      '423777003b835fc0fca998d57b7ddbc7e2cbab45f6b9b83f4a5cab1851574ee0'],
 'large_scale/knapPI_1_200_1000_1': [1967,
                                     'e6e04f76d7ec645739c43388f88c0fd34394d17b4c8e44e11665bdaf241833cd'],
 'large_scale/knapPI_1_5000_1000_1': [48974,
                                      '0c9149c9c903f00ec2b27597b1476e8bb17c22241782fd0254fd2addf1cf387e'],
 'large_scale/knapPI_1_500_1000_1': [4904,
                                     '8d832f2025615b0fe7ac4e6b7794e46619eb4c3b40e09a04cfccb1d5f37754b1'],
 'large_scale/knapPI_2_10000_1000_1': [98114,
                                       'ef58d8a1ba897bf7614054f77d303d535025e63b7ec08c884ea4d7eee3193f1e'],
 'large_scale/knapPI_2_1000_1000_1': [9840,
                                      '632ddcb26baa3e749ce284f41fcb530562cb25d78e5576cdae80c18d015ff703'],
 'large_scale/knapPI_2_100_1000_1': [984, '32d6ec5f8c52cdde2dc14e77c75a2f3c3dc13fdb8d7050488f9597d77f27fe6d'],
 'large_scale/knapPI_2_2000_1000_1': [19668,
                                      '915388ec15fbc02f81725a07d734421ff25860c96380d59a611b6f132b5e402f'],
 'large_scale/knapPI_2_200_1000_1': [1969,
                                     'e1facb717aa12fb73addc58d318ef5e5645e37ef0612e15f6f4ea71f1151dc18'],
 'large_scale/knapPI_2_5000_1000_1': [49106,
                                      '284b631395590ca940b16b31c9c84bc48be66436918bf86d7be43dbcd3495dc5'],
 'large_scale/knapPI_2_500_1000_1': [4914,
                                     '41eecf8b1a11909e6915030a2a1debe725641d021672883d49cbc43b8e0e7a58'],
 'large_scale/knapPI_3_10000_1000_1': [99967,
                                       '2f9a38fd9a54645ee66643bb3a0d981b50fe5ad38d935e0f9a8008765cf0a83a'],
 'large_scale/knapPI_3_1000_1000_1': [10007,
                                      'a7e4ac6a12fb7240a5c7faaf1bcba3ef60c16af010ff0a61ca08de14f0668035'],
 'large_scale/knapPI_3_100_1000_1': [1006,
                                     '269a660e44a44d0f680ce107c1df6ed33d5102a06a148410d17e3b99198c4b88'],
 'large_scale/knapPI_3_2000_1000_1': [19999,
                                      '42a45de1c97ce7e93a7a30ba7853fdf4627cc8ef935e1d89b250ca66b16bc7dd'],
 'large_scale/knapPI_3_200_1000_1': [2003,
                                     '4e71597ea32fb82ae51d5e54b0c6faa9ad218b5253bdb95e5f5460a1ccb0f951'],
 'large_scale/knapPI_3_5000_1000_1': [50012,
                                      '6278d41a431b35120c90a519b630346f2b9a32dde13896f3ee6515832fea73a9'],
 'large_scale/knapPI_3_500_1000_1': [5012,
                                     'e62a4945cd26bbef41ad679d2fb495c62c21cdc908b3d602f07be617e963794d'],
 'low-dimensional-optimum/f10_l-d_kp_20_879': [4,
                                               '46372791018924b8cbc444334300f85a211d2f29a56f2bb4890780b5983fc201'],
 'low-dimensional-optimum/f1_l-d_kp_10_269': [3,
                                              '9cfd3c755be26b4e1645918e2a64a26e3d851ede421e0b257f783b443bc443d1'],
 'low-dimensional-optimum/f2_l-d_kp_20_878': [4,
                                              'e39eef82f61b21e2e7f762fcc4307358f165757f2e77ec855d6992f7e0191932'],
 'low-dimensional-optimum/f3_l-d_kp_4_20': [2,
                                            '9f14025af0065b30e47e23ebb3b491d39ae8ed17d33739e5ff3827ffb3634953'],
 'low-dimensional-optimum/f4_l-d_kp_4_11': [2,
                                            '535fa30d7e25dd8a49f1536779734ec8286108d115da5045d77f3b4185d8f790'],
 'low-dimensional-optimum/f5_l-d_kp_15_375': [8,
                                              '777b1f18d8e39ae8d880a72fc428418d518fa6fb1fcd603f2388314ec77ddd28'],
 'low-dimensional-optimum/f6_l-d_kp_10_60': [2,
                                             '41cfc0d1f2d127b04555b7246d84019b4d27710a3f3aff6e7764375b1e06e05d'],
 'low-dimensional-optimum/f7_l-d_kp_7_50': [3,
                                            '3346f2bbf6c34bd2dbe28bd1bb657d0e9c37392a1d5ec9929e6a5df4763ddc2d'],
 'low-dimensional-optimum/f8_l-d_kp_23_10000': [4,
                                                'a29dc99a7700c4b86150bf9d2d1c79955b0bea378c732e25a214f84dc06c394f'],
 'low-dimensional-optimum/f9_l-d_kp_5_80': [3,
                                            '38d66d9692ac590000a91b03a88da1c88d51fab2b78f63171f553ecc551a0c6f'],
 'low-dimensional/f10_l-d_kp_20_879': [123,
                                       'cafc30f6e81607d4c28d9e331158c94921fcab29751d5778ef469ec76295ba36'],
 'low-dimensional/f1_l-d_kp_10_269': [62, '9a3d02795fdf17c6cc949c8d7929e6af77c89f146cdb64a876178ab1e3d030cd'],
 'low-dimensional/f2_l-d_kp_20_878': [123,
                                      '2b37bd30e0969570957122730201ddb04f6d881da33aa6852ba1b93a210dad54'],
 'low-dimensional/f3_l-d_kp_4_20': [23, '05cbe114f9c735320e84ba15bd37727d32fd2e1ff60685033d94805f44103fa9'],
 'low-dimensional/f4_l-d_kp_4_11': [23, '4d56e18ea18238633062d9bdf55240125cc1477a453296c5bbd44a342e9851d5'],
 'low-dimensional/f5_l-d_kp_15_375': [301,
                                      '33eb27a324b914dfb096e2d328bfc5209839fe35789a67b4fd5bb964125e01ed'],
 'low-dimensional/f6_l-d_kp_10_60': [57, '86aa342f2d35e999a8a3120fe0a0139d393bf736bf860825f29e085b5995b7b7'],
 'low-dimensional/f7_l-d_kp_7_50': [41, 'bb6d1b82fe7b5e2dea539a0afd07b98f7e02146b2d9ab7c058bd6255d79d03bd'],
 'low-dimensional/f8_l-d_kp_23_10000': [192,
                                        '114258d61f79779d42d5d6ab390752652ca03bcbfbac627e626148733386e565'],
 'low-dimensional/f9_l-d_kp_5_80': [33, 'd5d11869687a168b5310aa3d7222dd88370681b41b050dffcec0f014194437a4'],
 'vivaldi_JS4XqGDpHU.png': [13653, '8f189e338dd7e0af5397d6e81591647ae1f97964b21d7d3630b5077c8987f830'],
 'vivaldi_SzYI2Wp9Wn.png': [44669, 'cd380794c8b73867a84027d0410eebd38e0e7287fddb85d350a4c8fe2588f848'],
 'vivaldi_rt0rkxDG9n.png': [15534, '58d5dbc3630bc2f6f0967a0f4775d1d2d16c3390770da6c5cc06cf4adc873504']}

# sha256 of the sorted 'path\tsize\tsha256\n' rows.
INSTANCE_MANIFEST_SHA256 = '1c8cf6e34a79c939fc27bebdd74e840b15ca38cbd3317ef976ccbac47dbbf652'

# git tree object of instances_01_KP at PRE_STAGE_9_COMMIT.
INSTANCE_GIT_TREE = '9bce55551303bf2f4a78bee4f73e58cc2775df2b'

# ------------------------------------------------------------- location contract (Stage 9)

def _family(names, module):
    return {name: frozenset({"noisyvis.problems." + module}) for name in names}


# Where each definition must live. Until Checkpoint D this also allowed each definition's
# pre-Stage-9 module; from Checkpoint D only the Stage 9 canonical module is accepted.
ALLOWED_LOCATIONS = {
    **_family(PRE_STAGE_9_DEFINITIONS["FitnessFunctions.py"][2:7], "onemax"),
    **_family(("jump_fitness",), "jump"),
    **_family(PRE_STAGE_9_DEFINITIONS["FitnessFunctions.py"][8:20], "knapsack"),
    **_family(("rastrigin_eval", "birastrigin_eval", "ackley"), "continuous"),
    **_family(("bitflip_prior_noise",), "onemax"),
    **_family(("mean_weight[single-objective users]",), "knapsack"),
    **_family(("mean_weight[multi-objective users]",), "knapsack_mo"),
    **_family(PRE_STAGE_9_DEFINITIONS["multiobjectiveFunctions.py"][1:], "knapsack_mo"),
    **_family(("knap_violation",), "constraints"),
    **_family(PRE_STAGE_9_DEFINITIONS["ProblemScripts.py"], "instances"),
}

# ------------------------------------------------------------------------------ the probe

# Shared by the main probe and the temporary forwarder probe. It imports no compatibility module.
_PROBE_PRELUDE = r'''
import ast
import builtins
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import stat
import symtable
import sys
import textwrap
import types
from pathlib import Path

SOURCE_ROOT = Path(__SOURCE_ROOT__)
WORKSPACE = Path(__WORKSPACE__)
EVALUATORS = __EVALUATORS__

# The source tree under inspection comes first; for the test itself this is the workspace.
sys.path.insert(0, str(SOURCE_ROOT / "src"))

spec = importlib.util.spec_from_file_location("_harness_fence", __FENCE__)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(str(WORKSPACE))

spec = importlib.util.spec_from_file_location("_legacy_paths", __LEGACY_PATHS__)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

import random
import numpy as np
import yaml
from hydra._internal.utils import _locate


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def ast_sha256(obj):
    body = ast.parse(textwrap.dedent(inspect.getsource(obj))).body
    if len(body) != 1:
        raise AssertionError(f"getsource({obj!r}) parsed to {len(body)} statements")
    return hashlib.sha256(ast.dump(body[0]).encode()).hexdigest()


def canon(obj):
    """Type-preserving JSON form; numpy scalars are checked before the Python types they subclass."""
    if isinstance(obj, np.bool_):
        return {"numpy.bool_": bool(obj)}
    if isinstance(obj, np.integer):
        return {"numpy." + type(obj).__name__: str(int(obj))}
    if isinstance(obj, np.floating):
        return {"numpy." + type(obj).__name__: float(obj).hex()}
    if isinstance(obj, np.ndarray):
        return {"ndarray": str(obj.dtype), "shape": list(obj.shape),
                "data": [canon(x) for x in obj.reshape(-1).tolist()]}
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return {"int": str(obj)}
    if isinstance(obj, float):
        return {"float": obj.hex()}
    if isinstance(obj, tuple):
        return {"tuple": [canon(x) for x in obj]}
    if isinstance(obj, list):
        return {"list": [canon(x) for x in obj]}
    if isinstance(obj, dict):
        return {"dict": [[canon(k), canon(v)] for k, v in obj.items()]}
    return {"other": type(obj).__qualname__, "repr": repr(obj)}


def digest(obj):
    return sha256(json.dumps(obj, separators=(",", ":")))


def describe(fn):
    if fn is None:
        return None
    return {"name": getattr(fn, "__name__", None), "callable": callable(fn),
            "ast": ast_sha256(fn) if callable(fn) else None,
            "defaults": repr(fn.__defaults__), "kwdefaults": repr(fn.__kwdefaults__)}


def config_values(key):
    def walk(node, out):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and isinstance(v, str) and v:
                    out.append(v)
                else:
                    walk(v, out)
        elif isinstance(node, list):
            for item in node:
                walk(item, out)
    found = {}
    for where in ("configs", "tests/configs"):
        values = []
        for path in sorted((SOURCE_ROOT / where).rglob("*.yaml")):
            walk(yaml.safe_load(path.read_text()), values)
        found[where] = values
    return found
'''

# The main probe never imports a compatibility module, so it keeps working once Stage 12 deletes them.
_PROBE_MAIN = r'''
# Runtime code reaches the problems namespace through the runners' imports.
import noisyvis
import noisyvis.experiments.runner
import noisyvis.experiments.lon_runner as lon_runner
import noisyvis.algorithms

problems = sys.modules["noisyvis.problems"]
algorithms = sys.modules["noisyvis.algorithms"]
report = {"meta": {"noisyvis_file": str(Path(noisyvis.__file__).resolve()),
                   "problems_file": str(Path(problems.__file__).resolve())}}

holders = sorted(name for name, module in list(sys.modules.items())
                 if module is not None and name.split(".")[0] in ("noisyvis", "src")
                 and "_active_logger" in vars(module))
logger_holder = sys.modules[holders[0]] if len(holders) == 1 else None
report["meta"]["logger_holders"] = holders

# 1. configured fitness_fn names through the dynamic namespace
fitness_values = config_values("fitness_fn")
report["configured"] = {
    "counts": {where: len(values) for where, values in fitness_values.items()},
    "names": {name: describe(getattr(problems, name, None))
              for name in sorted({v for values in fitness_values.values() for v in values})},
}

# 2. every top-level definition in the problems package, parsed statically
class AnchorNormaliser(ast.NodeTransformer):
    """Stage 9 Checkpoint A: `_KNAPSACK_DIR + '/x/'` is the anchored form of the literal 'instances_01_KP/x/'.

    Rewriting it back lets the frozen pre-Stage-9 hashes pin every other statement of the loaders.
    """

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name) and node.left.id == "_KNAPSACK_DIR"
                and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
            return ast.copy_location(ast.Constant(value="instances_01_KP" + node.right.value), node)
        return node


definitions, anchor_normalised = [], []
for path in sorted((SOURCE_ROOT / "src" / "noisyvis" / "problems").glob("*.py")):
    if path.name == "__init__.py":
        continue
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raw = ast.dump(node)
            normalised = ast.dump(AnchorNormaliser().visit(node))
            if normalised != raw:
                anchor_normalised.append(node.name)
            definitions.append([type(node).__name__, node.name, hashlib.sha256(normalised.encode()).hexdigest()])
report["definitions"] = sorted(definitions)
report["anchor_normalised"] = sorted(anchor_normalised)

# 3. evaluator definitions and the globals each one reads
evaluators = {name: getattr(problems, name, None) for name in EVALUATORS}


def referenced_globals(fn):
    """Names the def statement (defaults included) and its body, nested functions too, read globally."""
    top = symtable.symtable(textwrap.dedent(inspect.getsource(fn)), "<probe>", "exec")
    names = {sym.get_name() for sym in top.get_symbols()
             if sym.is_referenced() and sym.get_name() != fn.__name__}
    stack = list(top.get_children())
    while stack:
        table = stack.pop()
        names.update(sym.get_name() for sym in table.get_symbols() if sym.is_global())
        stack.extend(table.get_children())
    return sorted(names)


def global_tag(fn, name):
    namespace = fn.__globals__
    if name in namespace:
        obj = namespace[name]
        if isinstance(obj, types.ModuleType):
            return {"kind": "module", "module": obj.__name__}
        if callable(obj):
            return {"kind": "function", "name": obj.__name__, "ast": ast_sha256(obj),
                    "is_noisyvis_algorithms_export": obj is getattr(algorithms, obj.__name__, None),
                    "is_logger_holder_function": logger_holder is not None
                    and obj is getattr(logger_holder, obj.__name__, None)}
        return {"kind": "value", "type": type(obj).__qualname__}
    if hasattr(builtins, name):
        return {"kind": "builtin"}
    return {"kind": "unresolved"}


report["evaluators"] = {
    name: None if fn is None else {
        **describe(fn),
        "globals": {g: global_tag(fn, g) for g in referenced_globals(fn)},
    }
    for name, fn in evaluators.items()
}

# 4. evaluator outputs, log records and RNG consumption on fixed inputs
LOADER = _locate("noisyvis.problems.instances.load_problem_KP")
KP_PID = "f1_l-d_kp_10_269"
_, KP_CAPACITY, _, _, _, KP_ITEMS, _ = LOADER(KP_PID)
INPUTS = {
    "bits20": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    "ones20": [1] * 20,
    "ones19": [1] * 19 + [0],
    "kp_feasible": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],     # weight 233 <= capacity 269
    "kp_infeasible": [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],   # weight 348 > capacity 269
    "vec2": [0.5, -1.25],
}
SEED = 20260917
SEEDS = (SEED, 1, 2)   # several seeds, so prior-noise cases reach both their noise and no-noise branches
KP = ("kp_feasible", "kp_infeasible")
CASES = [
    ("OneMax_fitness", ("bits20",), {"noise_intensity": 1.5}),
    ("OneMax_prior_bitflip_fitness", ("bits20",), {"noise_intensity": 5}),
    ("OneMax_prior_mult_bitflip_fitness", ("bits20",), {"noise_intensity": 3}),
    ("OneMax_prior_pq_bitwise_fitness", ("bits20",), {"noise_intensity": 4}),
    ("OneMax_prior_pq_bitwise_fitness", ("bits20",), {"noise_probability": 10, "noise_intensity": 4}),
    ("OneMax_prior_1q_bitwise_fitness", ("bits20",), {"noise_intensity": 4}),
    ("jump_fitness", ("bits20", "ones20", "ones19"), {"gap_size": 3, "noise_intensity": 0}),
    ("eval_ind_kp", KP, {}),
    ("eval_noisy_kp_v1_simple", KP, {"noise_intensity": 40}),
    ("eval_noisy_kp_v2_simple", KP, {"noise_intensity": 40}),
    ("eval_noisy_kp_v1", KP, {"noise_intensity": 0.5}),
    ("eval_noisy_kp_v1_penalty", KP, {"noise_intensity": 0.5}),
    ("eval_noisy_kp_v2", KP, {"noise_intensity": 0.5}),
    ("eval_noisy_kp_v2_penalty", KP, {"noise_intensity": 0.5}),
    ("eval_noisy_kp_v3", KP, {"noise_intensity": 0.5}),
    ("eval_noisy_kp_prior_bitflip", KP, {"noise_intensity": 5}),
    ("eval_noisy_kp_prior_mult_bitflip", KP, {"noise_intensity": 3}),
    ("eval_noisy_kp_pq_prior_bitwise", KP, {"noise_intensity": 4, "noise_probability": 5}),
    ("eval_noisy_kp_1q_prior_bitwise", KP, {"noise_intensity": 4}),
    ("rastrigin_eval", ("vec2",), {"noise_intensity": 0.7}),
    ("birastrigin_eval", ("vec2",), {"noise_intensity": 0.7}),
    ("ackley", ("vec2",), {"noise_intensity": 0.7}),
    ("eval_noisy_kp_v1_mo", KP, {"noise_intensity": 0.5, "noisy_objective": 0}),
    ("eval_noisy_kp_v1_mo", ("kp_infeasible",), {"noise_intensity": 0.5, "noisy_objective": 1, "penalty": 1}),
    ("eval_noisy_kp_v1_mo_violation", KP, {"noise_intensity": 0.5, "noisy_objective": 0}),
    ("eval_noisy_kp_v1_mo_violation", ("kp_infeasible",), {"noise_intensity": 0.5, "noisy_objective": 2, "penalty": 1}),
    ("countingOnesCountingZeros", ("bits20",), {"noise_intensity": 1.0, "noisy_objective": 0}),
]


class RecordingLogger:
    def __init__(self):
        self.calls = []

    def log_noisy_eval(self, *args, **kwargs):
        self.calls.append(canon([list(args), kwargs]))


def run_case(fn, input_key, kwargs, with_logger, seed):
    """One fully isolated evaluation: fresh inputs, fresh seeds, the logger cleared on both sides."""
    get_logger = evaluators["OneMax_fitness"].__globals__["get_active_logger"]
    holder = sys.modules[get_logger.__module__]
    individual = copy.deepcopy(INPUTS[input_key])
    individual_before = copy.deepcopy(individual)
    call_kwargs = copy.deepcopy(kwargs)
    if input_key.startswith("kp_"):
        call_kwargs["items_dict"] = copy.deepcopy(KP_ITEMS)
        call_kwargs["capacity"] = copy.deepcopy(KP_CAPACITY)
    kwargs_before = canon(call_kwargs)

    holder.clear_active_logger()
    recorder = RecordingLogger() if with_logger else None
    if recorder is not None:
        holder.set_active_logger(recorder)
    random.seed(seed)
    np.random.seed(seed)
    try:
        output, error = fn(individual, **call_kwargs), None
    except Exception as exc:  # noqa: BLE001 - the error is part of the pinned behaviour
        output, error = None, f"{type(exc).__name__}: {exc}"
    finally:
        random_state = sha256(repr(random.getstate()))
        numpy_state = digest(canon(np.random.get_state()))
        holder.clear_active_logger()

    return {"output": canon(output), "error": error, "random_state": random_state,
            "numpy_state": numpy_state, "log": None if recorder is None else recorder.calls,
            "individual_unchanged": canon(individual) == canon(individual_before),
            "kwargs_unchanged": canon(call_kwargs) == kwargs_before,
            "logger_cleared": get_logger() is None}


outputs = {}
for index, (name, input_keys, kwargs) in enumerate(CASES):
    for input_key in input_keys:
        for seed in SEEDS:
            for with_logger in (False, True):
                case_id = f"{index:02d}:{name}[{input_key}]seed={seed}{'+logger' if with_logger else ''}"
                fn = evaluators[name]
                outputs[case_id] = None if fn is None else run_case(fn, input_key, kwargs, with_logger, seed)
report["outputs"] = outputs

# 5/6. configured problem paths, through their canonical modules. Path-neutral: a value is counted and
# resolved by its canonical spelling, whichever spelling the config uses.
CONFIGURED_PATHS = {"ProblemScripts": ("load_problem_KP", "_target_"),
                    "ViolationFunctions": ("knap_violation", "violation_fn")}
canonical_paths = {}
for name, (anchor, key) in CONFIGURED_PATHS.items():
    prefix = legacy.LEGACY_TO_CANONICAL["src.problems." + name + "."]
    canonical = importlib.import_module(prefix[:-1])
    public = sorted(n for n in vars(canonical) if not n.startswith("_"))
    values = {where: [v for v in map(legacy.canonicalise, found) if v.startswith(prefix)]
              for where, found in config_values(key).items()}
    resolution = {}
    for value in sorted({v for found in values.values() for v in found}):
        attribute = value.rsplit(".", 1)[1]
        try:
            obj = _locate(value) if key == "_target_" else lon_runner._import_from_dotted(value)
        except Exception as exc:  # noqa: BLE001
            resolution[value] = "unresolvable: " + type(exc).__name__
            continue
        resolution[value] = "canonical" if obj is getattr(canonical, attribute, object()) else "different object"
    canonical_paths[name] = {
        "canonical_module": canonical.__name__,
        "anchor_defined_in_canonical": getattr(canonical, anchor).__module__ == canonical.__name__,
        "canonical_public": public,
        "package_exports_identical": {n: getattr(problems, n, None) is getattr(canonical, n) for n in public},
        "value_counts": {where: len(found) for where, found in values.items()},
        "resolution": resolution,
    }
report["canonical_paths"] = canonical_paths

# 7. loader outputs for every instance, through the config target path
candidates = [SOURCE_ROOT / "instances" / "knapsack", SOURCE_ROOT / "instances_01_KP"]
existing = [str(path.relative_to(SOURCE_ROOT)) for path in candidates if path.is_dir()]
instance_dir = SOURCE_ROOT / existing[0] if len(existing) == 1 else None
pids = [] if instance_dir is None else sorted(
    entry.name for sub in ("low-dimensional", "large_scale") for entry in (instance_dir / sub).iterdir())
stats_fn = _locate("noisyvis.problems.instances.get_knapsack_problem_stats")
correlation_fn = _locate("noisyvis.problems.instances.interpret_correlation")
def attempt(fn, *args):
    """(result, None) or (None, "ExceptionType: message"); an exception is pinned behaviour too."""
    try:
        return fn(*args), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def pinned(fn, *args):
    result, error = attempt(fn, *args)
    return {"error": error} if error is not None else {"digest": digest(canon(result))}


loaded = {pid: attempt(LOADER, pid) for pid in pids}
report["loader"] = {
    "pids": pids,
    "digests": {pid: {"error": error} if error is not None else {"digest": digest(canon(result))}
                for pid, (result, error) in loaded.items()},
    "shapes": {pid: None if error is not None else [int(result[0]), int(result[1])]
               for pid, (result, error) in loaded.items()},
    "missing_error": attempt(LOADER, "no_such_instance_pid")[1],
    "stats": {pid: pinned(stats_fn, pid)
              for pid in ("f1_l-d_kp_10_269", "f5_l-d_kp_15_375", "f8_l-d_kp_23_10000",
                          "knapPI_1_100_1000_1", "knapPI_3_10000_1000_1")},
    "stats_missing": canon(stats_fn("no_such_instance_pid")),
    "correlation": [correlation_fn(r) for r in (float("nan"), 0.5, -0.5, 0.05, 0.1, -0.1)],
}

# 8. knap_violation: the config-reachable module-level copy versus the copy nested in the MO evaluator
violation = _locate("noisyvis.problems.constraints.knap_violation")
mo_violation = evaluators["eval_noisy_kp_v1_mo_violation"]
nested = [node for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(mo_violation))))
          if isinstance(node, ast.FunctionDef) and node.name == "knap_violation"]
module_node = ast.parse(textwrap.dedent(inspect.getsource(violation))).body[0]


def nested_violation(key):
    random.seed(SEED)
    return canon(mo_violation(copy.deepcopy(INPUTS[key]), copy.deepcopy(KP_ITEMS), copy.deepcopy(KP_CAPACITY),
                              noise_intensity=0, noisy_objective=0, penalty=0)[1])


report["violation"] = {
    "module_level": {key: canon(violation(copy.deepcopy(INPUTS[key]), copy.deepcopy(KP_ITEMS),
                                          copy.deepcopy(KP_CAPACITY))) for key in KP},
    "nested_in_mo_evaluator": {key: nested_violation(key) for key in KP},
    "module_return": ast.unparse(module_node.body[-1]),
    "nested_return": [ast.unparse(node.body[-1]) for node in nested],
    "module_ast": hashlib.sha256(ast.dump(module_node).encode()).hexdigest(),
    "nested_ast": [hashlib.sha256(ast.dump(node).encode()).hexdigest() for node in nested],
}

# 9. the two mean_weight copies, as their users reach them
MEAN_WEIGHT_USERS = {
    "single-objective": ("eval_noisy_kp_v1", "eval_noisy_kp_v1_penalty", "eval_noisy_kp_v2",
                         "eval_noisy_kp_v2_penalty", "eval_noisy_kp_v3"),
    "multi-objective": ("eval_noisy_kp_v1_mo", "eval_noisy_kp_v1_mo_violation"),
}
copies = {group: [evaluators[user].__globals__.get("mean_weight") for user in users]
          for group, users in MEAN_WEIGHT_USERS.items()}
report["mean_weight"] = {
    group: {"users": list(MEAN_WEIGHT_USERS[group]),
            "one_object": all(obj is objs[0] for obj in objs),
            "ast": sorted({ast_sha256(obj) for obj in objs}),
            "value_on_kp10": canon(objs[0](copy.deepcopy(KP_ITEMS)))}
    for group, objs in copies.items()
}
report["mean_weight"]["copies_are_distinct_objects"] = (
    copies["single-objective"][0] is not copies["multi-objective"][0])

# 10. problem imports of the dashboard and of graph_builder.add_lon_nodes, read statically
def problem_imports(path, package, function=None):
    tree = ast.parse(path.read_text())
    if function is None:
        nodes = tree.body
    else:
        nodes = [n for n in ast.walk(next(f for f in tree.body
                                          if isinstance(f, ast.FunctionDef) and f.name == function))]
    found = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            parts = package.split(".")
            base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            module = node.module if node.level == 0 else ".".join([*base, node.module] if node.module else base)
            if module == "noisyvis.problems" or module.startswith("noisyvis.problems."):
                found.append((module, [alias.name for alias in node.names]))
    return found


def resolve_imports(found):
    explicit, star, errors = {}, set(), []
    for module_name, names in found:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        for name in names:
            if name == "*":
                star.update(n for n in vars(module) if not n.startswith("_"))
            elif not hasattr(module, name):
                errors.append(f"{module_name}.{name} missing")
            else:
                explicit[name] = getattr(module, name) is getattr(problems, name, None)
    return {"explicit_is_package_object": dict(sorted(explicit.items())), "star_names": sorted(star),
            "star_evaluators_are_package_objects": all(
                getattr(sys.modules[m], n) is getattr(problems, n)
                for m, names in found if "*" in names for n in vars(sys.modules[m]) if n in EVALUATORS),
            "errors": errors}


src_pkg = SOURCE_ROOT / "src" / "noisyvis"

# Stage 10 moved `add_lon_nodes` from visualization/graph_builder.py to the LON graph-population
# module viz/graph/lon.py. Checkpoint E tightened this to the post-move location only; the pins below
# are unchanged, because `problem_imports` resolves the relative import against the module's own
# package.
lon_path, lon_package = src_pkg / "viz" / "graph" / "lon.py", "noisyvis.viz.graph"
assert lon_path.is_file(), f"expected the LON graph-population module at {lon_path}"

# Stage 11 splits Dashboard.py across the dashboard package and removes its problem wildcards, so the
# dashboard site is the whole package rather than one file: the explicitly imported names must stay
# the same objects, and the star names must stop being imported without ever having been referenced.
def dashboard_problem_imports():
    found = []
    for path in sorted((src_pkg / "dashboard").rglob("*.py")):
        package = "noisyvis." + str(path.parent.relative_to(src_pkg)).replace("/", ".")
        found.extend(problem_imports(path, package.rstrip(".")))
    return found


def dashboard_referenced_names():
    """Names the dashboard reads that nothing in its own module binds: the wildcards' real payload.

    A name that the module imports explicitly or defines itself is bound whether or not a wildcard
    also injects it, so only these unbound reads would change meaning if a wildcard were removed.
    """
    unbound = set()
    for path in sorted((src_pkg / "dashboard").rglob("*.py")):
        tree = ast.parse(path.read_text())
        loads, bound = set(), set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                (loads if isinstance(node.ctx, ast.Load) else bound).add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
                spec = getattr(node, "args", None)
                if spec is not None:
                    bound.update(a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs)
                    if spec.vararg:
                        bound.add(spec.vararg.arg)
                    if spec.kwarg:
                        bound.add(spec.kwarg.arg)
            elif isinstance(node, ast.Lambda):
                spec = node.args
                bound.update(a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        unbound.update(loads - bound)
    return sorted(unbound)


report["imports"] = {
    "dashboard": resolve_imports(dashboard_problem_imports()),
    "graph_builder.add_lon_nodes": resolve_imports(problem_imports(
        lon_path, lon_package, function="add_lon_nodes")),
}
report["dashboard_referenced_names"] = dashboard_referenced_names()


# 11. the knapsack instance tree, byte for byte
def git_tree_sha1(path):
    entries = []
    for entry in os.scandir(path):
        if entry.is_symlink():
            data = os.readlink(entry.path).encode()
            mode, sort_name = b"120000", entry.name
            sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        elif entry.is_dir():
            mode, sort_name, sha = b"40000", entry.name + "/", git_tree_sha1(entry.path)
        else:
            data = Path(entry.path).read_bytes()
            mode = b"100755" if entry.stat().st_mode & stat.S_IXUSR else b"100644"
            sort_name = entry.name
            sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        entries.append((sort_name.encode(), mode + b" " + entry.name.encode() + b"\0" + bytes.fromhex(sha)))
    body = b"".join(item for _, item in sorted(entries))
    return hashlib.sha1(b"tree " + str(len(body)).encode() + b"\0" + body).hexdigest()


manifest, symlinks = {}, []
if instance_dir is not None:
    for dirpath, dirnames, filenames in os.walk(instance_dir):
        for name in dirnames + filenames:
            if os.path.islink(os.path.join(dirpath, name)):
                symlinks.append(os.path.relpath(os.path.join(dirpath, name), instance_dir))
        for name in filenames:
            full = Path(dirpath) / name
            data = full.read_bytes()
            manifest[full.relative_to(instance_dir).as_posix()] = [len(data), hashlib.sha256(data).hexdigest()]
rows = "".join(f"{path}\t{size}\t{sha}\n" for path, (size, sha) in sorted(manifest.items()))
report["instances"] = {
    "existing_locations": existing,
    "files": len(manifest),
    "bytes": sum(size for size, _ in manifest.values()),
    "symlinks": sorted(symlinks),
    "manifest_sha256": sha256(rows),
    "manifest": dict(sorted(manifest.items())),
    "git_tree": None if instance_dir is None else git_tree_sha1(instance_dir),
}

# 12. where each definition lives, found through the objects runtime code reaches
locations = {name: getattr(fn, "__module__", None) for name, fn in evaluators.items()}
bitflip_users = [evaluators[user].__globals__.get("bitflip_prior_noise")
                 for user in ("OneMax_prior_bitflip_fitness", "eval_noisy_kp_prior_bitflip")]
locations["bitflip_prior_noise"] = (bitflip_users[0].__module__
                                    if bitflip_users[0] is not None and bitflip_users[0] is bitflip_users[1]
                                    else "NOT ONE SHARED OBJECT")
locations["mean_weight[single-objective users]"] = copies["single-objective"][0].__module__
locations["mean_weight[multi-objective users]"] = copies["multi-objective"][0].__module__
locations["knap_violation"] = violation.__module__
for name in ("load_problem_KP", "get_knapsack_problem_stats", "interpret_correlation"):
    locations[name] = _locate("noisyvis.problems.instances." + name).__module__
report["locations"] = locations

print(json.dumps(report))
'''

_PROBE = _PROBE_PRELUDE + _PROBE_MAIN

# TEMPORARY (Stage 12): the `src.problems.*` forwarders, imported only here. Deleted with the
# forwarders in Stage 12 Checkpoint C.
_FORWARDER_PROBE_BODY = r'''
import noisyvis.experiments.lon_runner as lon_runner
import noisyvis.problems

problems = sys.modules["noisyvis.problems"]
FORWARDERS = {"ProblemScripts": ("load_problem_KP", "_target_"),
              "ViolationFunctions": ("knap_violation", "violation_fn")}
forwarders = {}
for name, (anchor, key) in FORWARDERS.items():
    forwarder = importlib.import_module("src.problems." + name)
    canonical = sys.modules[getattr(forwarder, anchor).__module__]
    public = sorted(n for n in vars(forwarder) if not n.startswith("_"))
    prefix = "src.problems." + name + "."
    canonical_prefix = legacy.LEGACY_TO_CANONICAL[prefix]
    # Every configured value of this family, in its legacy spelling, whichever spelling the config uses.
    values = {where: [prefix + v[len(canonical_prefix):] for v in map(legacy.canonicalise, found)
                      if v.startswith(canonical_prefix)]
              for where, found in config_values(key).items()}
    resolution = {}
    for value in sorted({v for found in values.values() for v in found}):
        attribute = value.rsplit(".", 1)[1]
        try:
            obj = _locate(value) if key == "_target_" else lon_runner._import_from_dotted(value)
        except Exception as exc:  # noqa: BLE001
            resolution[value] = "unresolvable: " + type(exc).__name__
            continue
        resolution[value] = ("canonical" if obj is getattr(canonical, attribute, object())
                             and obj is getattr(forwarder, attribute, object()) else "different object")
    forwarders[name] = {
        "file": str(Path(forwarder.__file__).resolve().relative_to(SOURCE_ROOT)),
        "canonical_module": canonical.__name__,
        "public": public,
        "canonical_public": sorted(n for n in vars(canonical) if not n.startswith("_")),
        "not_identical": [n for n in public if getattr(forwarder, n) is not getattr(canonical, n, object())],
        "package_exports_identical": {n: getattr(problems, n, None) is getattr(forwarder, n) for n in public},
        "value_counts": {where: len(found) for where, found in values.items()},
        "resolution": resolution,
    }

print(json.dumps(forwarders))
'''


def _substitute(template: str, source_root: Path) -> str:
    replacements = {
        "__SOURCE_ROOT__": repr(str(source_root)),
        "__WORKSPACE__": repr(str(WORKSPACE)),
        "__FENCE__": repr(str(HARNESS_DIR / "fence.py")),
        "__LEGACY_PATHS__": repr(str(HARNESS_DIR.parent / "legacy_paths.py")),
        "__EVALUATORS__": repr(EVALUATORS),
    }
    probe = template
    for marker, value in replacements.items():
        probe = probe.replace(marker, value)
    return probe


def build_probe(source_root: Path) -> str:
    """The probe for one source tree. The tests only ever pass the workspace."""
    return _substitute(_PROBE, source_root)


def build_forwarder_probe(source_root: Path) -> str:
    """TEMPORARY (Stage 12): the forwarder-only probe, deleted with the forwarders in Checkpoint C."""
    return _substitute(_PROBE_PRELUDE + _FORWARDER_PROBE_BODY, source_root)


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run the probe once in a harness subprocess (fresh temp root, fence installed)."""
    root = make_temp_root()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", build_probe(WORKSPACE)],
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, f"problems-package probe failed:\n{completed.stderr[-4000:]}"
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert report["meta"]["noisyvis_file"].startswith(str(WORKSPACE / "src" / "noisyvis")), report["meta"]
    return report


@pytest.fixture(scope="module")
def forwarder_probe() -> dict:
    """TEMPORARY (Stage 12): run the forwarder-only probe, deleted with the forwarders in Checkpoint C."""
    root = make_temp_root()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", build_forwarder_probe(WORKSPACE)],
            cwd=str(root),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, f"problems forwarder probe failed:\n{completed.stderr[-4000:]}"
        return json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _mismatches(observed: dict, expected: dict) -> list:
    keys = sorted(set(observed) | set(expected))
    return [f"{key}:\n    expected {expected.get(key)!r}\n    observed {observed.get(key)!r}"
            for key in keys if observed.get(key) != expected.get(key)]


# ------------------------------------------------------------------------------ the tests


def test_configured_fitness_fns_resolve_to_pinned_definitions(probe):
    configured = probe["configured"]

    assert configured["counts"]["configs"] > 0 and configured["counts"]["tests/configs"] > 0, configured["counts"]
    assert set(configured["names"]) == CONFIGURED_FITNESS_FNS, (
        f"configured fitness_fn names changed:\n"
        f"  missing: {sorted(CONFIGURED_FITNESS_FNS - set(configured['names']))}\n"
        f"  added:   {sorted(set(configured['names']) - CONFIGURED_FITNESS_FNS)}"
    )
    for name, found in configured["names"].items():
        assert found is not None, f"getattr(sys.modules['noisyvis.problems'], {name!r}) fails"
        assert found["callable"] and found["name"] == name, f"{name}: {found}"
        assert found["ast"] == EVALUATOR_AST[name], f"{name} resolves to a changed definition"
        assert [found["defaults"], found["kwdefaults"]] == EVALUATOR_DEFAULTS[name], f"{name} defaults changed"


def test_problem_definitions_pinned_and_unique(probe):
    observed = probe["definitions"]
    expected_names = sorted(name for names in PRE_STAGE_9_DEFINITIONS.values() for name in names)

    assert len(observed) == 31, f"{len(observed)} top-level problem definitions, expected 31"
    assert all(kind == "FunctionDef" for kind, _, _ in observed), observed
    assert sorted(name for _, name, _ in observed) == expected_names, (
        "problem definitions lost, duplicated or added; only mean_weight may appear twice"
    )
    changed = sorted({name for _, name, sha in observed if sha != DEFINITION_AST[name]})
    assert not changed, f"problem definitions changed since the pre-Stage-9 tree: {changed}"
    # Only the two loaders may differ from their pins, and only by the INSTANCES_DIR path anchoring.
    assert set(probe["anchor_normalised"]) <= {"load_problem_KP", "get_knapsack_problem_stats"}, (
        f"the instance-path normalisation touched unexpected definitions: {probe['anchor_normalised']}"
    )


def test_fitness_globals_closure_pinned(probe):
    assert len(probe["meta"]["logger_holders"]) == 1, probe["meta"]["logger_holders"]
    evaluators = probe["evaluators"]

    assert set(evaluators) == set(EVALUATORS) and len(EVALUATORS) == 24
    missing = sorted(name for name, found in evaluators.items() if found is None)
    assert not missing, f"not exported by noisyvis.problems: {missing}"
    for name, found in evaluators.items():
        assert found["ast"] == EVALUATOR_AST[name], f"{name} changed"
        assert [found["defaults"], found["kwdefaults"]] == EVALUATOR_DEFAULTS[name], f"{name} defaults changed"
        problems = _mismatches(found["globals"], EVALUATOR_GLOBALS[name])
        assert not problems, f"{name} reaches different globals:\n  " + "\n  ".join(problems)


def test_evaluator_outputs_and_rng_consumption_pinned(probe):
    outputs = probe["outputs"]

    covered = {case_id.split(":", 1)[1].split("[", 1)[0] for case_id in outputs}
    assert covered == set(EVALUATORS), f"cases do not cover every evaluator: {sorted(set(EVALUATORS) ^ covered)}"
    for case_id, record in outputs.items():
        assert record is not None, f"{case_id}: evaluator missing"
        assert record["logger_cleared"], f"{case_id}: active logger leaked out of the call"
    changed = {case_id: outputs.get(case_id)
               for case_id in sorted(set(outputs) | set(EVALUATOR_OUTPUTS))
               if case_id not in outputs or _record_digest(outputs[case_id]) != EVALUATOR_OUTPUTS.get(case_id)}
    assert not changed, (
        f"{len(changed)} evaluator case(s) changed output, log records or RNG consumption:\n"
        + "\n".join(f"  {case_id}: {json.dumps(record)[:600]}" for case_id, record in changed.items())
    )


def _record_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("forwarder", ["ProblemScripts", "ViolationFunctions"])
def test_problem_config_paths_resolve_to_canonical_objects(probe, forwarder):
    """Permanent: every configured problem path resolves in its canonical module, whichever spelling."""
    report = probe["canonical_paths"][forwarder]
    expected_module = LEGACY_TO_CANONICAL[f"src.problems.{forwarder}."][:-1]

    assert report["canonical_module"] == expected_module, report["canonical_module"]
    assert report["canonical_module"].startswith("noisyvis.problems."), report["canonical_module"]
    assert report["anchor_defined_in_canonical"], f"anchor is not defined in {report['canonical_module']}"

    expected = FORWARDER_PUBLIC_NAMES[forwarder]
    assert set(report["canonical_public"]) == expected, (
        f"{report['canonical_module']} public namespace differs from the frozen forwarder namespace:\n"
        f"  missing: {sorted(expected - set(report['canonical_public']))}\n"
        f"  added:   {sorted(set(report['canonical_public']) - expected)}"
    )
    assert all(report["package_exports_identical"].values()), (
        f"noisyvis.problems does not export the canonical objects: {report['package_exports_identical']}"
    )

    assert report["value_counts"]["configs"] > 0, "no config values found; the walk looks broken"
    assert report["value_counts"]["tests/configs"] > 0, "no test-config values found; the walk looks broken"
    assert report["resolution"] and set(report["resolution"].values()) == {"canonical"}, report["resolution"]


# TEMPORARY (Stage 12): deleted with the forwarders in Checkpoint C.
@pytest.mark.parametrize("forwarder", ["ProblemScripts", "ViolationFunctions"])
def test_problem_forwarder_preserves_namespace_and_identity(forwarder_probe, forwarder):
    report = forwarder_probe[forwarder]

    assert report["file"] == f"src/src/problems/{forwarder}.py"
    assert report["canonical_module"].startswith("noisyvis.problems."), report["canonical_module"]
    assert report["canonical_module"] != f"src.problems.{forwarder}"

    expected = FORWARDER_PUBLIC_NAMES[forwarder]
    assert set(report["public"]) == expected, (
        f"src.problems.{forwarder} namespace changed:\n"
        f"  missing: {sorted(expected - set(report['public']))}\n"
        f"  added:   {sorted(set(report['public']) - expected)}"
    )
    assert set(report["canonical_public"]) == expected, (
        f"{report['canonical_module']} public namespace differs from the frozen forwarder namespace"
    )
    assert not report["not_identical"], f"re-exports different objects for: {report['not_identical']}"
    assert all(report["package_exports_identical"].values()), (
        f"noisyvis.problems does not export the canonical objects: {report['package_exports_identical']}"
    )

    assert report["value_counts"]["configs"] > 0, "no config values found; the walk looks broken"
    assert report["value_counts"]["tests/configs"] > 0, "no test-config values found; the walk looks broken"
    assert report["resolution"] and set(report["resolution"].values()) == {"canonical"}, report["resolution"]


def test_loader_outputs_pinned_for_every_instance(probe):
    loader = probe["loader"]

    assert loader["pids"] == sorted(LOADER_DIGESTS) and len(LOADER_DIGESTS) == 31, "knapsack instance PIDs changed"
    assert loader["shapes"] == LOADER_SHAPES, "\n".join(_mismatches(loader["shapes"], LOADER_SHAPES))
    changed = sorted(pid for pid in LOADER_DIGESTS if loader["digests"].get(pid) != LOADER_DIGESTS[pid])
    assert not changed, f"load_problem_KP output changed for: {changed}"
    assert loader["missing_error"] == LOADER_MISSING_ERROR
    assert loader["stats"] == STATS_DIGESTS, _mismatches(loader["stats"], STATS_DIGESTS)
    assert loader["stats_missing"] is None
    assert loader["correlation"] == CORRELATION_LABELS


def test_knap_violation_divergence_preserved(probe):
    violation = probe["violation"]

    # Deliberately NOT equivalent (D7/B6): the module-level copy is signed, the nested copy is clamped.
    assert violation == VIOLATION_PINS, "\n".join(_mismatches(violation, VIOLATION_PINS))
    assert violation["module_level"]["kp_feasible"] != violation["nested_in_mo_evaluator"]["kp_feasible"]
    assert violation["module_ast"] not in violation["nested_ast"]


def test_mean_weight_copies_pinned(probe):
    mean_weight = probe["mean_weight"]

    assert mean_weight == MEAN_WEIGHT_PINS, "\n".join(_mismatches(mean_weight, MEAN_WEIGHT_PINS))
    # Equivalent implementations, kept as two distinct definitions.
    assert mean_weight["single-objective"]["ast"] == mean_weight["multi-objective"]["ast"]
    assert mean_weight["single-objective"]["value_on_kp10"] == mean_weight["multi-objective"]["value_on_kp10"]
    assert mean_weight["copies_are_distinct_objects"]


def test_visualization_problem_imports_resolve(probe):
    imports = probe["imports"]

    for site, expected in IMPORT_PINS.items():
        found = imports[site]
        assert not found["errors"], f"{site}: {found['errors']}"
        assert found["explicit_is_package_object"] == expected["explicit_is_package_object"], (
            f"{site}: explicitly imported names changed or are not the package's objects"
        )
        # Stage 11 removes the dashboard's four `problems` wildcards, which is the one intended
        # change here: amendment A1 keeps every explicit import, including the unused
        # `load_problem_KP`, so the set above is unchanged either way. From 11-I only the
        # post-removal state is accepted for the dashboard.
        allowed_star = [[]] if site == "dashboard" else [expected["star_names"]]
        assert found["star_names"] in allowed_star, (
            f"{site}: star-imported problems namespace changed:\n"
            f"  missing: {sorted(set(expected['star_names']) - set(found['star_names']))}\n"
            f"  added:   {sorted(set(found['star_names']) - set(expected['star_names']))}"
        )
        assert found["star_evaluators_are_package_objects"], f"{site}: star import provides non-canonical evaluators"

    # The wildcards are removable precisely because nothing reads what they inject: no name they
    # supplied is referenced anywhere in the dashboard package, before or after the removal.
    referenced = set(probe["dashboard_referenced_names"])
    leaked = sorted(set(IMPORT_PINS["dashboard"]["star_names"]) & referenced)
    assert not leaked, (
        "the dashboard reads names that only the `problems` wildcards supply, so removing them "
        f"would change resolution: {leaked}"
    )


def test_instance_tree_manifest_unchanged(probe):
    instances = probe["instances"]

    assert len(instances["existing_locations"]) == 1, (
        f"exactly one knapsack instance tree must exist: {instances['existing_locations']}"
    )
    assert instances["symlinks"] == []
    assert (instances["files"], instances["bytes"]) == (66, 631796), (instances["files"], instances["bytes"])
    changed = _mismatches(instances["manifest"], INSTANCE_MANIFEST)
    assert not changed, "instance files changed:\n  " + "\n  ".join(changed)
    assert instances["manifest_sha256"] == INSTANCE_MANIFEST_SHA256
    assert instances["git_tree"] == INSTANCE_GIT_TREE


def test_definition_family_locations(probe):
    locations = probe["locations"]

    assert set(locations) == set(ALLOWED_LOCATIONS) and len(ALLOWED_LOCATIONS) == 31
    misplaced = {label: module for label, module in locations.items() if module not in ALLOWED_LOCATIONS[label]}
    assert not misplaced, f"definitions outside their Stage 9 canonical module: {misplaced}"


# ------------------------------------------------------- loader anchoring (Stage 9 Checkpoint A)

ANCHOR_PIDS = ("f1_l-d_kp_10_269", "knapPI_1_100_1000_1")  # one low-dimensional, one large-scale instance

_ANCHOR_PROBE = r'''
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__SOURCE_ROOT__) / "src"))
spec = importlib.util.spec_from_file_location("_harness_fence", __FENCE__)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(__WORKSPACE__)

import numpy as np
from hydra._internal.utils import _locate

__CANON__

loader = _locate("noisyvis.problems.instances.load_problem_KP")
results = {}
for pid in __PIDS__:
    try:
        results[pid] = {"digest": digest(canon(loader(pid)))}
    except Exception as exc:  # noqa: BLE001
        results[pid] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps({"cwd": os.getcwd(), "cwd_entries": sorted(os.listdir(".")),
                  "noisyvis_root": os.environ.get("NOISYVIS_ROOT"), "results": results}))
'''


def build_anchor_probe(source_root: Path) -> str:
    """The loader-anchoring probe; `canon`/`digest` are the main probe's own definitions, verbatim."""
    import ast

    tree = ast.parse(_PROBE)
    helpers = "\n\n".join(ast.get_source_segment(_PROBE, node) for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name in ("sha256", "canon", "digest"))
    replacements = {
        "__SOURCE_ROOT__": repr(str(source_root)),
        "__WORKSPACE__": repr(str(WORKSPACE)),
        "__FENCE__": repr(str(HARNESS_DIR / "fence.py")),
        "__PIDS__": repr(ANCHOR_PIDS),
        "__CANON__": helpers,
    }
    probe = _ANCHOR_PROBE
    for marker, value in replacements.items():
        probe = probe.replace(marker, value)
    return probe


def test_loader_is_anchored_to_instances_dir(tmp_path):
    """The loader finds instances through NOISYVIS_ROOT (INSTANCES_DIR), whatever the working directory.

    Before Stage 9 Checkpoint A the loader resolved `instances_01_KP/...` against the cwd, so this run,
    from an empty directory unrelated to the project root, raised FileNotFoundError.
    """
    root = make_temp_root()                      # exposes <root>/instances -> /workspace/instances
    unrelated_cwd = tmp_path / "unrelated-cwd"   # empty, outside both the root and the workspace
    unrelated_cwd.mkdir()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", build_anchor_probe(WORKSPACE)],
            cwd=str(unrelated_cwd),
            env=child_env(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert completed.returncode == 0, f"anchoring probe failed:\n{completed.stderr[-4000:]}"
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert Path(report["cwd"]).resolve() == unrelated_cwd.resolve() and report["cwd_entries"] == []
    assert report["noisyvis_root"] == str(root) and not str(unrelated_cwd).startswith(str(root))
    assert report["results"] == {pid: LOADER_DIGESTS[pid] for pid in ANCHOR_PIDS}, report["results"]
