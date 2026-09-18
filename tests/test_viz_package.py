"""Visualisation and plotting contracts, characterised before the Stage 10 merge (plan Stage 10, Checkpoint 0).

Stage 10 merges `visualization/` and `plotting/` into `noisyvis.viz`, moves graph statistics to
`noisyvis.analysis.graph_stats`, splits `graph_builder.py` into the two graph-**population** modules
`viz/graph/stn.py` and `viz/graph/lon.py`, and moves the two `dash_table` builders to
`dashboard/components.py`. The reproducibility baselines never execute any of this code, and the
layering rules only scan imports. These tests pin what neither can see:

 1. every top-level definition of the 19 visualisation/plotting modules, by normalised AST and by its
    string-literal multiset, each existing exactly once and only in its pre- or intended post-move module;
 2. the plot registry: key order, callable identity, the Dashboard dropdown values that reach it, the
    live `plot2d_*` performance aliases, and the absence of the 12 camelCase Pareto aliases and the
    legacy monolith that Checkpoint F removed;
 3. every reachable Pareto plot with the Dashboard's own argument variants;
 4. both performance-plot families, including their missing-column fallbacks;
 5. the LON-stats figures, the graph-statistics correlations and both Dash table builders;
 6. the real `update_plot` orchestration over one shared graph (M1-M10), plus the nine-layout smoke
    matrix, at three levels: the graph immediately before layout, the returned positions, and the
    rendered outputs;
 7. the Dashboard, DashboardHelpers and layout/components bodies, imports excluded, so Stage 10 can be
    shown to change imports only;
 8. the visualisation names those three files import, resolved to the objects they name.

Locations are pinned through the facades and through a file scan. Checkpoints 0-D2 accepted both the
pre- and the post-move tree, so the same tests held at every intermediate step; **Checkpoint E set
ALLOW_PRE_LOCATIONS to False**, so only the final Stage 10 locations are accepted: `noisyvis.viz.*`,
`noisyvis.viz.graph.{stn,lon}`, `noisyvis.viz.plots.*`, `noisyvis.analysis.graph_stats` and
`noisyvis.dashboard.components`. The pre-move packages `noisyvis.visualization` and
`noisyvis.plotting` no longer exist: no source file may import them, and neither may be importable.
Only the location contract was tightened at E; every behavioural expectation below is still the value
frozen from PRE_STAGE_10.

The EXPECTED values are frozen literals captured from a `git archive` of the untouched PRE_STAGE_10
commit, under the runner's Python 3.11, in fresh subprocesses. They were never derived from the
implementation under test. AST hashes are `sha256(ast.dump(...))` after relative imports are rewritten
to absolute modules, so a definition keeps its hash when it moves to a deeper package; `ast.dump`
output depends on the Python version, so a runner upgrade legitimately needs them re-captured from
PRE_STAGE_10.

The probe runs in a harness subprocess with the write fence installed, so the pytest process never
imports the science packages and never imports `noisyvis.dashboard.Dashboard`, which would load the
1.3 GB warehouse. `update_plot` is instead executed from its own AST, with the real Dashboard import
block, helpers and constants, and with spy wrappers that delegate to the real functions.

PYTHONHASHSEED is pinned to 0 for the probe subprocess only. `Dashboard.update_plot` builds its
advanced-misjudgement traces by iterating a Python `set` of node-label strings (Dashboard.py:2234 and
2281), so the point order inside those traces alone follows string hash randomisation. That is a
pre-existing presentation-order nondeterminism, not a Stage 10 discrepancy. It is covered twice: the
exact figure digest under the pinned seed, and `figure_semantic`, which sorts the point tuples of the
advanced-misjudgement traces only. Nothing else is order-insensitive: graph nodes, graph edges,
positions, ordinary traces and callback outputs keep their exact iteration order, which is the I-10b
contract.

Run cost (measured on the runner): the `core` probe is about 3 s, `pareto` about 41 s and `heavy`
about 92 s. They are three independent subprocesses, one per fixture, so a targeted gate that selects
only the cheap tests pays only for `core`. See tests/README.md for the three gate groups.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from harness.run_isolated import HARNESS_DIR, WORKSPACE, child_env, make_temp_root

# The frozen PRE_STAGE_11 statement order of the two dashboard files, owned by the Stage-11
# characterization. Stage 11 moves those statements, so the bodies below are rebuilt from it rather
# than read from one file; the expected BODIES hashes are unchanged (plan §15.1).
from test_dashboard_package import STATEMENTS as _STAGE_11_STATEMENTS

PRE_STAGE_10_COMMIT = "4530785ccae448365800cce88bb96a556d98e75a"
BODY_KEYS = {key: _STAGE_11_STATEMENTS[key]
             for key in ("dashboard/Dashboard.py", "dashboard/DashboardHelpers.py")}

# Checkpoint E tightened this to False: the pre-move locations under `visualization/` and `plotting/`
# are no longer accepted, so every definition must live at its final Stage-10 location.
ALLOW_PRE_LOCATIONS = False

# --------------------------------------------------------------- frozen at PRE_STAGE_10
DEFINITIONS = {'AXIS_LABELS': {'ast': '00451c08df9142812bfad4b670d56466a3ad5c898d460b9edea53b91e8ca3505',
                 'kind': 'const',
                 'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c'},
 'AXIS_OPTIONS': {'ast': 'd70e18c5739a80286ce8be90f61c00040e9cbee68b598cb8165474c87dd691f7',
                  'kind': 'const',
                  'literals': '8f5eeaf14acac59fef7c524b0ed194362d97b91e812b9ad5f1f79a938792de12'},
 'AxisConfig': {'ast': 'eb99fba5e42c28def1f1c24b1845f3d14237e48156da9f9e3e5304871cd7ed28',
                'kind': 'class',
                'literals': '029e818ebb322c67c5b25b57e66971158c13ca27399113144274b341643d1089'},
 'CameraConfig': {'ast': 'b38fe18404c77f0591078be58d2c52f76f16ec6e4d59cad22ef79a07f7fdbfbf',
                  'kind': 'class',
                  'literals': 'fe8af41fa63ed275874907d8ad2449f343e6b966f8b6be59ffe1b079f70dbd0b'},
 'DEFAULT_COLORSCALE': {'ast': '59c080e90e519099dc99037276a1f0872e869d4e2360fd27ea7f6c8be2d793aa',
                        'kind': 'const',
                        'literals': '2a8886a88dc6c00080a77a7110c78f6c2676968c4442fd2c317afa9a29548ad6'},
 'DEFAULT_LINE_WIDTH': {'ast': '5b5b927ceb6e099214f2eecfa9a270581aeacc150ec37e9226033bbe1f4dae6e',
                        'kind': 'const',
                        'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c'},
 'DEFAULT_MARKER_SIZE': {'ast': '76b12b5fbddaca56a61f945e0ba2dc20d6448591714f63ff2f103178defbe3e5',
                         'kind': 'const',
                         'literals': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c'},
 'DEFAULT_PLOT_STYLE': {'ast': 'de0e5f84729e07d3e41af5e1ce16df4dfd48abc7bff5240c6e2e2486bed868f4',
                        'kind': 'const',
                        'literals': '788ab640b93668d885d43129c49ccf174e97163b7ad67af565d214579c42488c'},
 'DEFAULT_TEMPLATE': {'ast': '14af14bde811901d3de60cd3ff8d53ebc87729a7cb2d918a7adaa08134a1d782',
                      'kind': 'const',
                      'literals': 'd748a07a6f44241f19ebdeacdf774a8d13a004fcf082edc6e9f6410878b1d330'},
 'DEFAULT_X_AXIS': {'ast': '9f1a7d89512f39880990ab2761ddd5fe8dbfb62c9d6c1676aabc24d49add16b2',
                    'kind': 'const',
                    'literals': '6e6d36eb9605cd9adbafe83455b4b4bd031c9e91ce6ea0bb50978326c3c6cf2e'},
 'DEFAULT_Y_AXIS': {'ast': '5b74257d72cecdbef6c275cad13f7b47326def164fef112145ae96341549bbbb',
                    'kind': 'const',
                    'literals': '59d380661f798a7563ea17019213332801cf2e2db27c7af9be259e3148bf6cf6'},
 'LONConfig': {'ast': '0ad66a9ecd6fe1c5dd4f8d98324e8d641f9a0eaacc2a192f5f7ca19184a16135',
               'kind': 'class',
               'literals': 'b0974c5f3f0fc91f9451576e5b343a3cbefa212fd060a9a695672aeef6218d96'},
 'LONStatistics': {'ast': '6e96666e0a89663f1623738568219f92da30a7f5545a92e1d808ddde4cb66965',
                   'kind': 'class',
                   'literals': '087c405554225938cfa446e5e38c82e2706a588577a37db3bde8bb21dec3d1e3'},
 'NodeSizeConfig': {'ast': '176056bfad702c70af073a384589984bdc1c851e3c9ec19b8691c56d1d0dcbb4',
                    'kind': 'class',
                    'literals': 'e568d336eb8d5c7f16f4c5c9a10e3216aef53599e7a26a1f17f678d08aa74ad7'},
 'NoisyLONConfig': {'ast': '10e943dda3193048dae6e56c64a9b729619faf860b9e484ca6b904b1a9a46a10',
                    'kind': 'class',
                    'literals': 'd0d2977f666434835b1efcd9573a2f14946909f4258c1129d6299f151367dc34'},
 'OpacityConfig': {'ast': '213b6c76c0d800fb4ee49ae3283b4d8c2ebf45e5c8aa0d8e597d4474e43e5783',
                   'kind': 'class',
                   'literals': '599b9b97d1d1354b8349b30a3a994fd39380eb0da62dec3ab12b24af029d14f0'},
 'PARETO_COLORSCALE': {'ast': '4ecf7e2f3e339b6a8899700e02cb80417dee6247f7f2bf72576f77adbcfccc07',
                       'kind': 'const',
                       'literals': 'a00d63c21452e1481ab0fbf453e545d7b80eff2a1c4fbb07c03ba509100d2bbd'},
 'PARETO_PLOTS': {'ast': '75697902bbeb507f5af4ce8739069e5e3cc434d23816241c0c89ac45b3d55d98',
                  'kind': 'const',
                  'literals': '4a7c2463760b5efcbdbfaaf4f145560d41c361d875a90f896cd509de2e0e0709'},
 'PERFORMANCE_PLOTS': {'ast': '4b3aa193440f0a88a4420b62a78138fafe505ce1864912b398a0bc1b465a1c0d',
                       'kind': 'const',
                       'literals': '8a445e3d1be24316e24360817b37a0e0070e5c75b325f7a5044b64846f271a4c'},
 'PLOT_STYLE_OPTIONS': {'ast': '36dae368b0e91861516a96c12779e0813331a61bac4906197d112ec12a31213a',
                        'kind': 'const',
                        'literals': '05bef94139f3aaa4c09c6c270adef69248f3a18e48d9b46877610eef12d06580'},
 'PlotConfig': {'ast': '6508fdf1b8c1a8380eb6b51862dc52a0b873a50202e01e9bac4c76dde75bf903',
                'kind': 'class',
                'literals': '51e298d7182911ea93f2bbad44af2440a1fcc82cd947f99d0484a02712e15c0b'},
 'STNConfig': {'ast': '2999887b80bcd1403999895f7031c92644906ef182371e976390f79a57b936cb',
               'kind': 'class',
               'literals': '3521c39cfd41623970d6cbe9c9b75ffe1108a2d562948b1d2d5ebc59ec427d08'},
 '_GUIDE_COLORS': {'ast': 'a5dbb2b95301998ad8c0ba210bb8c4ffffdacc504fbca8d6af8d01ae8a69b39a',
                   'kind': 'const',
                   'literals': '38224d3bbdcb7f3df7ac61d80681a37fbbb491c05d56d9a705ff78299afd8b21'},
 '_GUIDE_NAMES': {'ast': '92783a69b3885e527753010951576453cbcd6f2e1212731dd6bd36391c735cde',
                  'kind': 'const',
                  'literals': '1ade131f54260d083591945d42e01601811f4c4324dae64606c0bfc8846e1bc2'},
 '_is_dual_front': {'args': ['G', 'noisy_nodes'],
                    'ast': '95ca1ae2bd380383463fb262871a72c7e87ce38ba5bd07de35d9fa999b04b5ba',
                    'kind': 'def',
                    'literals': '0a21990ddad99e521e7f4abc1dec8f58ef68e10f66958f2bcef89c48225ce486'},
 '_pearson_r': {'args': ['x', 'y'],
                'ast': '8697ec55c007cdb8f1645de22c54c64d2a6abc235ec8e5d02d8b00239f0b3ca1',
                'kind': 'def',
                'literals': '8da4d1e6e04041938ebbef9922227c38d69a2bd02b217fa56a7a591063c24c86'},
 '_precalculate_stn_edge_colors': {'args': ['G'],
                                   'ast': 'e1d1f15263d2bf254c7ebc15ad4b6b5a78229f05ce6703a4a8200525e3acec8f',
                                   'kind': 'def',
                                   'literals': '8f28eafc2342dccfb513f1f35841f6f1a33935caf54415ec3a21785c048d9145'},
 '_spearman_r': {'args': ['x', 'y'],
                 'ast': '76cb894b97095c8e413c22dfd2e0100402f509185cd55590832da04d866e8a3f',
                 'kind': 'def',
                 'literals': '0289b13b07419742a0e0684073201b7287d2a2e16f175032f13d0995c625fbd0'},
 '_viridis_colors': {'args': ['n', 'colorscale'],
                     'ast': '0deab57d8a963d0f46801c528c41a7f249d60730cd2611052b4f85e8f1ae2c9f',
                     'kind': 'def',
                     'literals': '2a8886a88dc6c00080a77a7110c78f6c2676968c4442fd2c317afa9a29548ad6'},
 'add_lon_edges': {'args': ['G', 'local_optima', 'lon_node_mapping', 'config', 'opt_feas_map'],
                   'ast': '5c6f73c6f71997c13e9b1452614e546adb34cf291956e851d25b1963ddd040d1',
                   'kind': 'def',
                   'literals': '9eb8d30b0a7c1c231103082343ddfcb83421b0920049c8ce936d21880c83857d'},
 'add_lon_nodes': {'args': ['G',
                            'local_optima',
                            'lon_node_mapping',
                            'config',
                            'problem_id',
                            'fitness_func_params'],
                   'ast': '16374f54f260ab1e2fee43f985834ce62309305b8cc8fbcd9fde6a709a5a1eb5',
                   'kind': 'def',
                   'literals': 'bc10d7c3b84bcbeb2fb705749d6819f35a37597735c5587cf8470b355871787f'},
 'add_mo_fronts': {'args': ['G', 'mo_runs_for_series', 'edge_color', 'series_idx', 'noisy_node_color'],
                   'ast': 'd7d09e7be956b3ab28c5c69c410b4d3c508d8916c6dd009e84730d0953f9d788',
                   'kind': 'def',
                   'literals': 'dabc474cb81ad7f1c586bf576302989320fb11c95aecc816f8396be80bf6ab93'},
 'add_prior_noise_stn_algo_pov': {'args': ['G',
                                           'all_run_trajectories',
                                           'edge_color',
                                           'series_idx',
                                           'noisy_node_color',
                                           'dedup',
                                           'show_alt_rep',
                                           'show_alt_rep_no_fit',
                                           'stn_node_min',
                                           'use_est_discarded_as_base'],
                                  'ast': '62496cb4794186ccf7b1fbd987e708bd40e2fa8fd2bfab25f0871667c0befcf8',
                                  'kind': 'def',
                                  'literals': 'c068b067b041c32f5858402e93ba843828e26404420c571e470792e11e5bc655'},
 'add_prior_noise_stn_v4': {'args': ['G',
                                     'all_run_trajectories',
                                     'edge_color',
                                     'series_idx',
                                     'noisy_node_color',
                                     'dedup',
                                     'show_alt_rep',
                                     'show_alt_rep_no_fit',
                                     'stn_node_min',
                                     'use_est_discarded_as_base'],
                            'ast': 'c42e2f94f099b62623e1462a2b54f6fc956db9b4f5536412afac9556ec1486d6',
                            'kind': 'def',
                            'literals': '245947aef832cbaf4ab4ea4eb43a3cdc77dc40493901f1be2d299c18715825c4'},
 'add_prior_noise_stn_v5': {'args': ['G',
                                     'all_run_trajectories',
                                     'edge_color',
                                     'series_idx',
                                     'noisy_node_color',
                                     'dedup',
                                     'show_alt_rep',
                                     'show_alt_rep_no_fit',
                                     'stn_node_min',
                                     'use_est_discarded_as_base'],
                            'ast': '4fd049f24ea1e06140d2f54103162df757a9b10482244b7cfa3997729104a01a',
                            'kind': 'def',
                            'literals': 'c504563b3c9c235ff599aa17a1181af52ac74d028668c1a2a8c8c8f497e03c4b'},
 'add_stn_trajectories': {'args': ['G',
                                   'all_run_trajectories',
                                   'edge_color',
                                   'algo_idx',
                                   'stn_node_mapping',
                                   'config'],
                          'ast': '18818678708be3d6a31954bf43fadd7d453843d32a390c773116911fcd9c318b',
                          'kind': 'def',
                          'literals': 'e8b6aac56c6b9136f197548fcac385d4afd6de6f0424d8490b01fdd2cfc4923f'},
 'apply_generation_coloring': {'args': ['G', 'colorscale'],
                               'ast': '44f6661694f5bbbe2f22003bed67dc03b5df8e477b4c061678457a56e4a9e352',
                               'kind': 'def',
                               'literals': '7ded27bee33403c3c4405c93409ff1111df255ea0bd881e556f152f0b474ccb2'},
 'apply_node_colors': {'args': ['G', 'config', 'opt_feas_map', 'neigh_feas_map', 'optimum'],
                       'ast': '4a2b27d62b23cfd0bcb6d3f78db7dfd57eae2163081a78bccb17783bd37dab08',
                       'kind': 'def',
                       'literals': 'de570d69ba26e5f54bd9616dc5018749b6d595bf05debb9a27cdc0b99b1e6599'},
 'apply_node_sizes': {'args': ['G', 'config', 'optimum', 'visit_prop_map'],
                      'ast': 'c720add70f9c2fc52cca53270725925333fafa07da9fc06c25a33fc7018898b3',
                      'kind': 'def',
                      'literals': 'ddba2cbecabb7260c6112cf16ac12770b94ed3a344901bf69ca95f9d976ee28b'},
 'apply_standard_layout': {'args': ['fig', 'title', 'xaxis_title', 'yaxis_title', 'legend_title', 'height'],
                           'ast': '82fd228122656e6d33efbd2c94f5b39e92ba722f856da0d227d2acb4ad34927d',
                           'kind': 'def',
                           'literals': '891f18bd72d8d82167d3b9934e13022a5d75672c085d1a579053830daa94f1e4'},
 'build_all_traces': {'args': ['G', 'pos', 'config', 'node_noise', 'fitness_dict', 'neigh_feas_map'],
                      'ast': 'bbf7a780d64370f7afd256b5936c88108af496985a4fd6621068f61359bff168',
                      'kind': 'def',
                      'literals': 'bc082dac929d5866620f600239d87bb465113287bd75587b2fba817f8bfe5ccd'},
 'build_correlation_table': {'args': ['correlations'],
                             'ast': 'a470beefdd4ef2dba56e076be9b3c67db8df1b0f9071bac568d6a3fc657090f5',
                             'kind': 'def',
                             'literals': '461d7013b0333850b68ec91155b3c7f028f8d11cd434cf56378a3f0f92606548'},
 'build_selected_correlation_display': {'args': ['correlation', 'x_label', 'y_label'],
                                        'ast': '48fbfa39552765bff9d0c3d72e9c4019914853101e68add1ce530f40f2308cb3',
                                        'kind': 'def',
                                        'literals': 'ea0a873a00306d48cf263b72378b9ef873418f50d7b69eac2c5b4768135a70b7'},
 'calculate_graph_summary': {'args': ['G', 'verbose'],
                             'ast': 'e292536f84fa9b5d10ebf5d9f5008ef0705abda379669c6f02fed2f8826a957c',
                             'kind': 'def',
                             'literals': '0fd2b15c94385088385dc3709de718b5cb78762f14576cd97c8513dcf926f6e8'},
 'calculate_lon_statistics': {'args': ['G', 'verbose'],
                              'ast': '6560bdcd253d53dac15d3376894ba6c60d0049f0bed6b4c495ae69f7e4b04af0',
                              'kind': 'def',
                              'literals': '081fbb860f61fdd9874638abce9da510b38a2bf2a8d3ca290d4b68089ac471e4'},
 'calculate_positions': {'args': ['G', 'layout_type', 'stn_plot_type', 'plot_3d', 'lmds_multiplier'],
                         'ast': '860cd1cb5ad743ff17302e125b20070fcae3363c6b31ca0d4090f42325a55de3',
                         'kind': 'def',
                         'literals': 'b495afa0543da518e73b309df174c1eb236ab0bd5699f1e0b1c125342a5ccfb4'},
 'calculate_positions_mo': {'args': ['G', 'layout_type'],
                            'ast': '2d2342f979a4ebfc9e3f734fc7e6e2922c977458319f555a96c9bc19571452f5',
                            'kind': 'def',
                            'literals': 'ac289086cb70a8bce3dcd7dcf917e99c4a518c3f4cfb25283025cd71b99010a4'},
 'calculate_positions_so': {'args': ['G', 'layout_type', 'plot_3d', 'lmds_multiplier'],
                            'ast': 'fb0b03a8e4915a59cd7f08432cbc221160140770c7e0744dc1a28d1bd17b8eb3',
                            'kind': 'def',
                            'literals': '47801130ba2b9ab4180a11b4c857af35fdd52d4de689bb86dd4858e17abde57c'},
 'calculate_stn_statistics': {'args': ['G', 'verbose'],
                              'ast': '2a4a721a72190eb73f9b37d5c6678be607e3579d633aecae94aec48ba97a5756',
                              'kind': 'def',
                              'literals': 'e8f82f4bb4045eaea5dcd4573a7bfd29a1cc33a74ab995478a3f48830f8cb96e'},
 'compute_correlation_pair': {'args': ['node_stats', 'x_key', 'y_key'],
                              'ast': 'f1f64f65296abc44db293a60823598354ac8709865b8049cf503e0297de26546',
                              'kind': 'def',
                              'literals': 'a4c457f1ab5998c605f7e40db08d7d27f55698b2b15e443c482d1164610a94d5'},
 'compute_generation_range': {'args': ['gen_entries'],
                              'ast': '895726c9201e6118c3c263ef96f9dbedcc21f313c66458c695fdcd2bb16ad9ae',
                              'kind': 'def',
                              'literals': '31fe5e0eccf02bcfb0fc9a730a8a518d21ecb5b8c6f64252b4c64ed4cb78ecec'},
 'compute_node_feasibility_error': {'args': ['G', 'pos', 'node_noise', 'fitness_dict', 'neigh_feas_map'],
                                    'ast': '33f16584712f6f4cc8657512d9f97c22e28c68dd7fc3eaa46af9efc24a22f7d2',
                                    'kind': 'def',
                                    'literals': '6b992233c3ec2c1b17607811d47448f337874249e53f02634463dcc6eae1f4f5'},
 'compute_pairwise_correlations': {'args': ['node_stats'],
                                   'ast': '289a9f3d3dc61a22f85e907cea87064c94623b22acc3ad08cd4dc8dfd5657501',
                                   'kind': 'def',
                                   'literals': '27ce8cecc4e4d8fd51a77ebecdddf385ada58b9b70ef2d4bf9fe6e6e70a5c3dd'},
 'create_axis_settings': {'args': ['G', 'pos', 'config', 'node_noise', 'axes_text_scale'],
                          'ast': '7db8637c06d7e280bb577779ddab3db65160061a748bc419160fde3176eeea83',
                          'kind': 'def',
                          'literals': '3dcd69fffd60e0af72a78d780299be919ff462367918470c40e558698a2219f3'},
 'create_boxplot_traces': {'args': ['pos', 'node_noise', 'fitness_dict', 'config', 'iqr_only'],
                           'ast': '2d71dfb5d37f68d396fbb1e5901c58521439368e17701c1f1c8fb8fa6931cc14',
                           'kind': 'def',
                           'literals': '7d77198a33a9dba3ca9a5d9c3276aedd9550b3b53d2346eec5bfd88bc1a34cd2'},
 'create_edge_label_trace': {'args': ['edge_label_x', 'edge_label_y', 'edge_label_z', 'edge_labels'],
                             'ast': 'd9d63ac88cea54c4e69b311300f693bbc9112ba52a5c59bbfa8dcf1c06ab7511',
                             'kind': 'def',
                             'literals': '278439780a4f0dfb3d2b6e00c2a6bc4e37a98095b52fe55c24f2d05dbff789e3'},
 'create_edge_traces': {'args': ['G', 'pos', 'config'],
                        'ast': '893393ca74034d73a425719968f5019158adc50d11fd009348c61606c6ffc40f',
                        'kind': 'def',
                        'literals': '2106cd59347c6c99ed54c60d4bd3aed3af6003aae662d4358ed6f2de3a9e303a'},
 'create_empty_figure': {'args': ['title'],
                         'ast': '54f395099e0a2bc30d96a7495fa3606f5645e97415071c788e0eab006f798a1c',
                         'kind': 'def',
                         'literals': '5ec8e55fd3ecd376e8164a65dfa13c1acab4c3cdfe6bb13f2a0988926c2b25f2'},
 'create_estimated_fitness_traces': {'args': ['G', 'pos', 'config'],
                                     'ast': 'e16b7981d6342d59653f9d111bad698e5a2df06c185fe51dafeb7e20de5a3b4d',
                                     'kind': 'def',
                                     'literals': '1be7d64037d29003f9762d4222117cd98ac2e3cd411261683f721d752000f3b2'},
 'create_evals_colorbar_trace': {'args': ['eval_min', 'eval_max', 'colorscale'],
                                 'ast': '81649a841936dd8a5ab0dad465ab3d431140c044e27774c24b72e9fee090c090',
                                 'kind': 'def',
                                 'literals': '07937c5c951f9751b184383f0911d0303a56e1f2420fc9e637b293ec7270386e'},
 'create_figure': {'args': ['traces',
                            'config',
                            'xaxis_settings',
                            'yaxis_settings',
                            'zaxis_settings',
                            'output_path',
                            'scene_annotations'],
                   'ast': 'b9f77986cfda392064269e7e1c1cbce066c4fc02dd904bb0ba5f037f5e46154b',
                   'kind': 'def',
                   'literals': 'bc1b06f99b51b46beeef9859776ae53c07fd54330b0ce5dac5ad595079d71b40'},
 'create_guide_traces': {'args': ['G', 'pos'],
                         'ast': '4b601205206e5b59e9d278af57be55e15271acede44ecd99e61144fecf06b8b8',
                         'kind': 'def',
                         'literals': '154592562f261d5dd0013a4c97d8e1899ecab8a0cd78ddc0d9e41de7e447b6a9'},
 'create_hover_text': {'args': ['G', 'hover_info_value'],
                       'ast': 'd86c88cafbf662a4c234ba0deb914522ec92956194ec92d9d60586a38b064a1a',
                       'kind': 'def',
                       'literals': '312e1577d6da87b5ba4568db07fac9598dfa598d4c26770901fc4e3180092541'},
 'create_lon_colorbar_trace': {'args': ['cmin', 'cmax', 'title', 'x_pos', 'colorscale'],
                               'ast': 'ba19a8886fb469cbdf58fbdf929e78347af4d4477ec6f67c6e78e84d2e8975ec',
                               'kind': 'def',
                               'literals': '9736881be8bb72d29baf06a6b19690a4818fc66b01e696a950928948f28b9187'},
 'create_lon_mesh_trace': {'args': ['x', 'y', 'z', 'colour_values', 'opacity'],
                           'ast': '26900a10667259d9acbdc4478a7f574c817512b3d96e172b7d1db33008badbdb',
                           'kind': 'def',
                           'literals': 'fac765ec96f6c30750f7e637066ce62352cf8cb23702adf4e9962fa4d478ae0a'},
 'create_lon_surface_trace': {'args': ['x',
                                       'y',
                                       'z',
                                       'colour_values',
                                       'opacity',
                                       'grid_resolution',
                                       'colorscale',
                                       'cmin',
                                       'cmax'],
                              'ast': 'a45dd6bdb4009e7d0373fab71280fcf43d18a4ca0f57f2edc7da7ee412bbd6a8',
                              'kind': 'def',
                              'literals': '1b66d8726ef4bb0949a3434854f05459bcffc371f820855459d36d3ccb95b81f'},
 'create_node_traces': {'args': ['G', 'pos', 'config'],
                        'ast': '85af99566abc1205ab12a4ba316d0af9eb69a6ae6e1a54dfed12ac25542011c6',
                        'kind': 'def',
                        'literals': 'fee220f1b7068110a399cb76780dedfe3b766e7a5106907f471a839595df33b1'},
 'create_pareto_plot': {'args': ['plot_type', 'frontdata', 'series_labels'],
                        'ast': '7da0c67976e9e528d8d7dac68d792849b3cf8f46f5e42420de0d5a33650ba8e0',
                        'kind': 'def',
                        'literals': 'd3928fee537b36b11b1294cd250578735a882ea9def424827d9b759fa152c404'},
 'create_performance_plot': {'args': ['plot_type', 'dataframe'],
                             'ast': '8b829e1e6413b6545c1637a3bd89864e19a15c2fd72fe55dc97b7c64acebf412',
                             'kind': 'def',
                             'literals': 'd7a6e4f0bfb3741cc36cc2e8711b8052a4124040d839a7af48ac44ba7c62082d'},
 'debug_mo_counts': {'args': ['G', 'by', 'label', 'list_fronts', 'max_list'],
                     'ast': 'c463074b4ea8f62d2554f140c1bcc73a9bc26978d1e09a864f69746a73ec976c',
                     'kind': 'def',
                     'literals': 'f8ee9feb6079ce403e198b0ff45b40cbaeb049a306a8935a794de854f6e540f1'},
 'generate_run_summary_string': {'args': ['selected_trajectories'],
                                 'ast': '6c7689af454a79cd0988ba32b2bd5c4fa524ef38ea318fe60edc6405de0b4f29',
                                 'kind': 'def',
                                 'literals': '18626aa2323647d3bc63e04bd692a5a23a92b93b6ecb6b781dc4a996bbf2a81e'},
 'generation_color': {'args': ['generation', 'gen_min', 'gen_max', 'scale'],
                      'ast': 'd39a5989b466f5ad2118897b0a472bca965aabedc3c33eb2ab370f4c8b78208d',
                      'kind': 'def',
                      'literals': 'e32d047d2094248abef3c9e446f9b250d5930b0e9c215b9f68dc1d8d32fd8cda'},
 'get_pareto_plot': {'args': ['plot_type'],
                     'ast': '8a660705f72a757524625733cf214685af2dbe09d7c2bf1d58825c804ef4b1fc',
                     'kind': 'def',
                     'literals': '69aa09ae32988cb5626445cf6a267fd3ffa065519f61456e13a0df0a5d0afe68'},
 'get_performance_plot': {'args': ['plot_type'],
                          'ast': 'a79df3925c27098d5d22ac90a613748baecd69da513b1144e2cf72e174012239',
                          'kind': 'def',
                          'literals': 'b8a0820eab5e3cf2795c2ec22e0a93b5b269355f1ac4160841598436f6190485'},
 'get_run_entries': {'args': ['runs_full', 'run_idx'],
                     'ast': '2c8fc279216a3aa25ad593027c59b0c3d6c6932ff98ea037b7e4487d382497f9',
                     'kind': 'def',
                     'literals': '5b1981144882973a0c7286b711fdf57efc53074fd37a2c3b14d5f894010d427a'},
 'get_series_info': {'args': ['frontdata', 'series_labels', 'group_idx'],
                     'ast': 'e7592290e5268e376902e98b1c21b0b84c786a324fa39efdfdef0ef3db6f2c7f',
                     'kind': 'def',
                     'literals': 'efa5211cf6f5ca0b0dc35db1b7f58e57d83249828984067003ec07d5701d4462'},
 'list_pareto_plot_types': {'args': [],
                            'ast': '0b33a3c30da13a4ec695c383fe59b8d19410d55f2b3d1497e9035412165210f1',
                            'kind': 'def',
                            'literals': '687aa589d001226e4bc006ea63f1a12abdb0051504387ceccb96a487b6366c99'},
 'list_performance_plot_types': {'args': [],
                                 'ast': '4b9c72a35c1746b5a9e96aa75378869ea72f94a6a1d59e6e95b5e73f57bcf009',
                                 'kind': 'def',
                                 'literals': 'c4ba8238147553aa5657fd9403e3200ef793bfae5fa064c2859930dd0395d0e8'},
 'parse_callback_inputs': {'args': ['optimum',
                                    'pid',
                                    'opt_goal',
                                    'options',
                                    'run_options',
                                    'stn_lower_fit_limit',
                                    'lo_fit_percent',
                                    'lon_options',
                                    'lon_node_colour_mode',
                                    'lon_surface_colour',
                                    'lon_edge_colour_feas',
                                    'lmds_multiplier',
                                    'nlon_fit_func',
                                    'nlon_intensity',
                                    'nlon_samples',
                                    'nlon_penalty',
                                    'layout_value',
                                    'plot_type',
                                    'hover_info_value',
                                    'azimuth_deg',
                                    'elevation_deg',
                                    'run_start_index',
                                    'n_runs_display',
                                    'axis_values',
                                    'opacity_noise_bar',
                                    'lon_node_opacity',
                                    'lon_edge_opacity',
                                    'stn_node_opacity',
                                    'stn_edge_opacity',
                                    'stn_node_min',
                                    'stn_node_max',
                                    'lon_node_min',
                                    'lon_node_max',
                                    'lon_edge_size_slider',
                                    'stn_edge_size_slider',
                                    'stn_plot_type',
                                    'node_size_metric',
                                    'colorscale'],
                           'ast': 'b3211930c9b4e6d78f5c6ef5b39b6793911b31be5b3150fcb3f1e7e6d533d899',
                           'kind': 'def',
                           'literals': '100f57c97ae2b9c4e1dd88ecf09cca3a9947654666cdecd966e8198a09626bd1'},
 'plot_animation': {'args': ['frontdata', 'series_labels'],
                    'ast': '0b6e4b7120591c2df0bb0c51b0564cc953de317342eaa9d30c6cfcf8341de189',
                    'kind': 'def',
                    'literals': '5818339257f78ef0dd4f7db839a427ef695392f99d0f0d3ad66d9d0fdaefe0f7'},
 'plot_basic': {'args': ['frontdata', 'series_labels', 'continuous_colorscale', 'colorscale'],
                'ast': '2f68804e7368b1a3250c7bf5a53c547b928fdad4b104472a7d3472406f290e10',
                'kind': 'def',
                'literals': 'e7204f72e1730f08585b121b26cf1ad7d773ab3e3db4133801b87eff148ea794'},
 'plot_box': {'args': ['dataframe', 'fitness_mode', 'problem_goal', 'xaxis_title', 'colorscale'],
              'ast': '404cad1d7cb32fd7003e9806b1b5306b55107e61c6d91d2d216b0c6403ebab83',
              'kind': 'def',
              'literals': '2b66fca6a879bcc5f69f1cb4fef4cefd5009fe48de271d63d516518e6df701f8'},
 'plot_box_advanced_misjudgements_so': {'args': ['dataframe', 'algo_name', 'xaxis_title', 'colorscale'],
                                        'ast': 'ee1f1585217b35d9553d4ef450ebde063fef616af2f551a8e3704d82298744ea',
                                        'kind': 'def',
                                        'literals': 'ad2daf7f4fc7d7559270e9c3d5f526c45d26d5d4a60164b37747342a3e7217ac'},
 'plot_box_evals': {'args': ['dataframe', 'fitness_mode', 'xaxis_title', 'colorscale'],
                    'ast': '7f72d7aadff43076dde88caf850035a44a4c705e413d47e7dc6e816e1aa2ca64',
                    'kind': 'def',
                    'literals': 'd32a1c45a3a493095ad0a7a3fa2f88d73b5b2c1323b1a0690bc9f56de555d73f'},
 'plot_box_misjudgements_so': {'args': ['dataframe', 'xaxis_title', 'colorscale'],
                               'ast': 'fc101959c79d3912dd47c02ab8acddf501817187ae6c4f3c686f19d5adb9c454',
                               'kind': 'def',
                               'literals': '50a44c4e7d89ec114bc29ec29cea71c2102006b6f582d45351d497e04888fdee'},
 'plot_box_mo': {'args': ['dataframe', 'colorscale'],
                 'ast': '0455279f7a4c88b5b0bd3b95293ad16b9f4d321473706414d8f00720f7779dee',
                 'kind': 'def',
                 'literals': '8c4c5c91f7c0a9ccdaea2ccfe459d37c97abe024e0d9acbe095feea713665665'},
 'plot_box_penalty': {'args': ['dataframe', 'fitness_mode', 'problem_goal', 'xaxis_title', 'colorscale'],
                      'ast': 'bf3d58160ec51aa00f33134129d525d0d6a967886430b31f8964547e95834357',
                      'kind': 'def',
                      'literals': 'dd2ce6e2ba3c7e2dd37c5ca2f3a3788c9973ace998a5369912e3848aa107066c'},
 'plot_igd_vs_dist': {'args': ['frontdata', 'series_labels', 'distance_method', 'nruns'],
                      'ast': '97a3c69fd0f4348af42abad1d286584ff01d81a0fb0628586cdc6c435894dfd1',
                      'kind': 'def',
                      'literals': 'dab2532e77318662c0c3ee4a978edccc60272a66f0329c21b318669c41856ec6'},
 'plot_ind_vs_dist': {'args': ['frontdata', 'series_labels', 'distance_method', 'nruns'],
                      'ast': '0b492122dd7e138c9389e5e78eecdad1d1cb836d14ed3aadf7dfabaf56a89fcf',
                      'kind': 'def',
                      'literals': 'a407eec03ff7e4d133ea582192d0d622904ad76e89861b0011ce7e91662e4a7a'},
 'plot_line': {'args': ['dataframe', 'fitness_mode', 'problem_goal', 'xaxis_title', 'colorscale'],
               'ast': 'cafd6966fcb48402fa63c19ab77c6a0c2ac9979e4ec3541a306cfd6493c68ef0',
               'kind': 'def',
               'literals': '4fe8b0e264f0b6c0fac6163c662e8353a0d63cde6cdf82308c4d6d8a2ee94cea'},
 'plot_line_evals': {'args': ['dataframe', 'fitness_mode', 'show_std', 'xaxis_title', 'colorscale'],
                     'ast': 'eb0485a5ef0edbe43c3361dd5af2a3a71d9cb517217be904fe6f349791ff4158',
                     'kind': 'def',
                     'literals': '262c330dc28306bf3e985f0cb6cc263e91e544adb42f48f165b42e08b70e8077'},
 'plot_line_mo': {'args': ['dataframe', 'colorscale'],
                  'ast': 'd244a841dbe682f342bb92294ac1a234096e16992270a1b6f86f16554a1c0632',
                  'kind': 'def',
                  'literals': 'f83b0faa61c9c4f5cba7485e06befc0bcc3aeaa96e560ffeefb39c17122f6323'},
 'plot_lon_scatter': {'args': ['node_stats', 'x_key', 'y_key'],
                      'ast': '86c0b29988712fb1757a9fa10acc7841437f2964c721b202b512dc00ffe180f4',
                      'kind': 'def',
                      'literals': '91004e1f90a300d678a37101b061b4d7e74359e324aac02ae7e7c3639f5f0dad'},
 'plot_lon_scatter_multi': {'args': ['node_stats_by_series', 'x_key', 'y_key'],
                            'ast': 'b4f091f8f279630776b614e4cf9268f72965430bb2d835afb0208318311b41e1',
                            'kind': 'def',
                            'literals': '6aab4c8eecffa097c54c0687f92830f1c4af58e69d72d3fe65b62ef087746463'},
 'plot_lon_stats': {'args': ['node_stats', 'x_key', 'y_key', 'plot_style'],
                    'ast': '3cf45315991737c2e0bd23c2cd733afa000a14add3f4cff505279f49cb749dc8',
                    'kind': 'def',
                    'literals': '9c5b9f12c0c9bdd99ee72bf4f0e963fcb60ead2f861eeb168257319545fd1587'},
 'plot_lon_stats_multi': {'args': ['node_stats_by_series', 'x_key', 'y_key', 'plot_style'],
                          'ast': '272fbb0b7a5c90b3a7d2c47f04ce75d02f78cc7ff509abd6a942579048040815',
                          'kind': 'def',
                          'literals': '9f3158b60b49d95aa0ebd12a4caacf44bcce8b40745ff374e7af9515da8fab9b'},
 'plot_lon_violin': {'args': ['node_stats', 'x_key', 'y_key'],
                     'ast': 'eefdae195dc5aa6e3680ac724351de44d1640691f72d4ded59a1e6d43ba0d49a',
                     'kind': 'def',
                     'literals': '7e672a0633f251d1f5559476e4a549c3effd1b4943c815968622ec14985eb308'},
 'plot_lon_violin_multi': {'args': ['node_stats_by_series', 'x_key', 'y_key'],
                           'ast': '658252f88a68651794dbaa16cb9ae6bcfecff4c800294a9f233d45c8b845dcce',
                           'kind': 'def',
                           'literals': '15b91c55bdab9aadb07c7191844e919661f5ab532037c8c49cc52a58dd88b1b3'},
 'plot_move_delta_histograms': {'args': ['frontdata',
                                         'series_labels',
                                         'group_idx',
                                         'solution_key',
                                         'IndVsDist_IndType',
                                         'bins_decision',
                                         'bins_objective',
                                         'include_zero_moves'],
                                'ast': '0880d2471e94dec484c1e7f2fba320e599de0a2039e07921ab221e8ea8f78d49',
                                'kind': 'def',
                                'literals': '8f0d5b3918f46e72751b6ff19a513da6045888b6c07dcb318f470982c2170f43'},
 'plot_movement_correlation': {'args': ['frontdata',
                                        'series_labels',
                                        'group_idx',
                                        'run_idx',
                                        'solution_key',
                                        'IndVsDist_IndType',
                                        'window',
                                        'corr_method',
                                        'show_deltas'],
                               'ast': '15796867a31fab3b3acd19f98009ecd5e0adc9a90b13bf112c73c3f0592d420c',
                               'kind': 'def',
                               'literals': '7a90274f5d5ba8690eada7df7b1fb66f9df6c210887fc4b0f4a8335d482f6e15'},
 'plot_noisy': {'args': ['frontdata', 'series_labels'],
                'ast': 'fce09008551c562f71973f06947c47f5a52d7c628b9d0ba87dea523b1ebcd76a',
                'kind': 'def',
                'literals': 'c11219d44b2622415050d1646c07991758b59bcd500441dff26fc945df9072fb'},
 'plot_objective_vs_decision': {'args': ['frontdata',
                                         'series_labels',
                                         'group_idx',
                                         'run_idx',
                                         'solution_key',
                                         'IndVsDist_IndType',
                                         'include_zero_moves',
                                         'color_by',
                                         'marker_size'],
                                'ast': 'db728dca1f2b25eb1d14ed2da821acb7abfe954e566d621129695c3fa8a83329',
                                'kind': 'def',
                                'literals': '0b8b7c66ba924c8a231800ad9c3d942ef907e83be49c3a3cdcd898168296e25f'},
 'plot_progress_per_movement': {'args': ['frontdata',
                                         'series_labels',
                                         'group_idx',
                                         'run_idx',
                                         'solution_key',
                                         'hv_key',
                                         'eps',
                                         'k_patience',
                                         'use_ratio',
                                         'show_deltas'],
                                'ast': 'bd36bce52e3806bc2eab5ad84ad9bf6f1e6eb422bbc030e308d1a82ba04f6d9e',
                                'kind': 'def',
                                'literals': '24642cae0b3da79a8b0a2c298637444fc3721f7183296ed5c35b651e01eb75fb'},
 'plot_subplots': {'args': ['frontdata', 'series_labels'],
                   'ast': 'fee65baba41485419e95a807fe8e2c0b27caad653b0b8c1ce3bf7f4510fbabbc',
                   'kind': 'def',
                   'literals': 'e709c68760b61aa5de045c83e927638500af2737d7310b18c5b31f5a2247695c'},
 'plot_subplots_highlighted': {'args': ['frontdata',
                                        'series_labels',
                                        'solutions_key',
                                        'fits_key',
                                        'dist_decimals'],
                               'ast': '1b37c149ac22b3ce11b491958346ea65d37957332de7be419c5383fdf34c7470',
                               'kind': 'def',
                               'literals': 'cd95fcfe825a1e05bc6ca64db4e481d2e909eefa9c955ffda193f1ad985bcbe4'},
 'plot_subplots_multi': {'args': ['frontdata', 'series_labels', 'nruns'],
                         'ast': '17654de1ae41434200b5ccf99d32d2316533b6f3eef02eafe51b4d7a4e71bd5c',
                         'kind': 'def',
                         'literals': 'db22c226e742c5b04ee9275a40e7d60822673084e0d0f95482d25e632529f73a'},
 'print_hamming_transitions': {'args': ['all_run_trajectories', 'print_sols', 'print_transitions'],
                               'ast': '2f6c747422757291108ab8a60be9c8fb866e41372dd6bea27f7f640ad5820d8b',
                               'kind': 'def',
                               'literals': '99aa04f795f5ead10d492285e5b19acb72bb2264894940b66b4920f83d176b16'},
 'style_nodes': {'args': ['G', 'config', 'opt_feas_map', 'neigh_feas_map', 'visit_prop_map'],
                 'ast': 'c5c2d640448f6ab7dd351810746c9c323f54cd8182a301433bf482903152f02b',
                 'kind': 'def',
                 'literals': '73cfcbc5ecac91cefc5aa86b7b63178895845c91f55f29e2b8ad81bc1b6fb589'},
 'symmetric_range': {'args': ['arr', 'pad'],
                     'ast': 'ea5520c5735907eed2799ac7ffafdedb5456e3db104fddf49bc5c06257e70b3b',
                     'kind': 'def',
                     'literals': '070efcda8c78c739459dbe9cd6061d657128bccd941d5f96bb6d56a10841b358'}}

DEFINITION_MODULES = {'AXIS_LABELS': 'visualization/lon_stats_plots.py',
 'AXIS_OPTIONS': 'visualization/lon_stats_plots.py',
 'AxisConfig': 'visualization/config.py',
 'CameraConfig': 'visualization/config.py',
 'DEFAULT_COLORSCALE': 'plotting/base.py',
 'DEFAULT_LINE_WIDTH': 'plotting/base.py',
 'DEFAULT_MARKER_SIZE': 'plotting/base.py',
 'DEFAULT_PLOT_STYLE': 'visualization/lon_stats_plots.py',
 'DEFAULT_TEMPLATE': 'plotting/base.py',
 'DEFAULT_X_AXIS': 'visualization/lon_stats_plots.py',
 'DEFAULT_Y_AXIS': 'visualization/lon_stats_plots.py',
 'LONConfig': 'visualization/config.py',
 'LONStatistics': 'visualization/statistics.py',
 'NodeSizeConfig': 'visualization/config.py',
 'NoisyLONConfig': 'visualization/config.py',
 'OpacityConfig': 'visualization/config.py',
 'PARETO_COLORSCALE': 'plotting/base.py',
 'PARETO_PLOTS': 'plotting/registry.py',
 'PERFORMANCE_PLOTS': 'plotting/registry.py',
 'PLOT_STYLE_OPTIONS': 'visualization/lon_stats_plots.py',
 'PlotConfig': 'visualization/config.py',
 'STNConfig': 'visualization/config.py',
 '_GUIDE_COLORS': 'visualization/trace_builder.py',
 '_GUIDE_NAMES': 'visualization/trace_builder.py',
 '_is_dual_front': 'visualization/node_positioning.py',
 '_pearson_r': 'visualization/statistics.py',
 '_precalculate_stn_edge_colors': 'visualization/trace_builder.py',
 '_spearman_r': 'visualization/statistics.py',
 '_viridis_colors': 'plotting/performance/line_plots.py',
 'add_lon_edges': 'visualization/graph_builder.py',
 'add_lon_nodes': 'visualization/graph_builder.py',
 'add_mo_fronts': 'visualization/graph_builder.py',
 'add_prior_noise_stn_algo_pov': 'visualization/graph_builder.py',
 'add_prior_noise_stn_v4': 'visualization/graph_builder.py',
 'add_prior_noise_stn_v5': 'visualization/graph_builder.py',
 'add_stn_trajectories': 'visualization/graph_builder.py',
 'apply_generation_coloring': 'visualization/node_styling.py',
 'apply_node_colors': 'visualization/node_styling.py',
 'apply_node_sizes': 'visualization/node_styling.py',
 'apply_standard_layout': 'plotting/base.py',
 'build_all_traces': 'visualization/trace_builder.py',
 'build_correlation_table': 'visualization/lon_stats_plots.py',
 'build_selected_correlation_display': 'visualization/lon_stats_plots.py',
 'calculate_graph_summary': 'visualization/statistics.py',
 'calculate_lon_statistics': 'visualization/statistics.py',
 'calculate_positions': 'visualization/node_positioning.py',
 'calculate_positions_mo': 'visualization/node_positioning.py',
 'calculate_positions_so': 'visualization/node_positioning.py',
 'calculate_stn_statistics': 'visualization/statistics.py',
 'compute_correlation_pair': 'visualization/statistics.py',
 'compute_generation_range': 'plotting/base.py',
 'compute_node_feasibility_error': 'visualization/statistics.py',
 'compute_pairwise_correlations': 'visualization/statistics.py',
 'create_axis_settings': 'visualization/trace_builder.py',
 'create_boxplot_traces': 'visualization/trace_builder.py',
 'create_edge_label_trace': 'visualization/trace_builder.py',
 'create_edge_traces': 'visualization/trace_builder.py',
 'create_empty_figure': 'plotting/base.py',
 'create_estimated_fitness_traces': 'visualization/trace_builder.py',
 'create_evals_colorbar_trace': 'visualization/trace_builder.py',
 'create_figure': 'visualization/trace_builder.py',
 'create_guide_traces': 'visualization/trace_builder.py',
 'create_hover_text': 'visualization/node_positioning.py',
 'create_lon_colorbar_trace': 'visualization/trace_builder.py',
 'create_lon_mesh_trace': 'visualization/trace_builder.py',
 'create_lon_surface_trace': 'visualization/trace_builder.py',
 'create_node_traces': 'visualization/trace_builder.py',
 'create_pareto_plot': 'plotting/registry.py',
 'create_performance_plot': 'plotting/registry.py',
 'debug_mo_counts': 'visualization/graph_builder.py',
 'generate_run_summary_string': 'visualization/graph_builder.py',
 'generation_color': 'plotting/base.py',
 'get_pareto_plot': 'plotting/registry.py',
 'get_performance_plot': 'plotting/registry.py',
 'get_run_entries': 'plotting/base.py',
 'get_series_info': 'plotting/base.py',
 'list_pareto_plot_types': 'plotting/registry.py',
 'list_performance_plot_types': 'plotting/registry.py',
 'parse_callback_inputs': 'visualization/config.py',
 'plot_animation': 'plotting/pareto/animation.py',
 'plot_basic': 'plotting/pareto/basic.py',
 'plot_box': 'plotting/performance/box_plots.py',
 'plot_box_advanced_misjudgements_so': 'plotting/performance/box_plots.py',
 'plot_box_evals': 'plotting/performance/box_plots.py',
 'plot_box_misjudgements_so': 'plotting/performance/box_plots.py',
 'plot_box_mo': 'plotting/performance/box_plots.py',
 'plot_box_penalty': 'plotting/performance/box_plots.py',
 'plot_igd_vs_dist': 'plotting/pareto/analysis.py',
 'plot_ind_vs_dist': 'plotting/pareto/analysis.py',
 'plot_line': 'plotting/performance/line_plots.py',
 'plot_line_evals': 'plotting/performance/line_plots.py',
 'plot_line_mo': 'plotting/performance/line_plots.py',
 'plot_lon_scatter': 'visualization/lon_stats_plots.py',
 'plot_lon_scatter_multi': 'visualization/lon_stats_plots.py',
 'plot_lon_stats': 'visualization/lon_stats_plots.py',
 'plot_lon_stats_multi': 'visualization/lon_stats_plots.py',
 'plot_lon_violin': 'visualization/lon_stats_plots.py',
 'plot_lon_violin_multi': 'visualization/lon_stats_plots.py',
 'plot_move_delta_histograms': 'plotting/pareto/correlation.py',
 'plot_movement_correlation': 'plotting/pareto/correlation.py',
 'plot_noisy': 'plotting/pareto/noisy.py',
 'plot_objective_vs_decision': 'plotting/pareto/correlation.py',
 'plot_progress_per_movement': 'plotting/pareto/analysis.py',
 'plot_subplots': 'plotting/pareto/subplots.py',
 'plot_subplots_highlighted': 'plotting/pareto/subplots.py',
 'plot_subplots_multi': 'plotting/pareto/subplots.py',
 'print_hamming_transitions': 'visualization/graph_builder.py',
 'style_nodes': 'visualization/node_styling.py',
 'symmetric_range': 'plotting/base.py'}

# `_viridis_colors` is deliberately defined twice, once in each performance module. The pair is pinned
# by its count and by the two module basenames; each copy is allowed only at its exact pre- or
# post-move package location. This is not a generic duplicate exemption: any other duplicated name, or
# either copy at a third location, fails.
DUPLICATE_DEFINITIONS = {"_viridis_colors": 2}

ALLOWED_DUPLICATE_MODULES = {
    "_viridis_colors": (
        "viz/plots/performance/box_plots.py", "viz/plots/performance/line_plots.py",
    ),
}

DUPLICATE_BASENAMES = {"_viridis_colors": ["box_plots.py", "line_plots.py"]}

# The legacy Pareto monolith, excluded from the inventory until Checkpoint F deletes it.
# Checkpoint F deleted the legacy Pareto monolith, so no module may be excluded from the inventory
# any more.
ALLOWED_MONOLITHS = ()

# The 12 camelCase Pareto aliases Checkpoint F removed, in their exact historical spelling, with the
# canonical function each one pointed at. Both must stay gone; the targets must stay reachable.
REMOVED_PARETO_ALIASES = {
    "plotParetoFront": "plot_basic",
    "plotParetoFrontSubplots": "plot_subplots",
    "plotParetoFrontSubplotsMulti": "plot_subplots_multi",
    "PlotparetoFrontSubplotsHighlighted": "plot_subplots_highlighted",
    "plotParetoFrontAnimation": "plot_animation",
    "plotParetoFrontNoisy": "plot_noisy",
    "plotParetoFrontIndVsDist": "plot_ind_vs_dist",
    "plotParetoFrontIGDVsDist": "plot_igd_vs_dist",
    "plotProgressPerMovementRatio": "plot_progress_per_movement",
    "plotMovementCorrelation": "plot_movement_correlation",
    "plotMoveDeltaHistograms": "plot_move_delta_histograms",
    "plotObjectiveVsDecisionScatter": "plot_objective_vs_decision",
}

# The pre-move packages, which Checkpoint E removed. They must not be importable and must not be
# imported by any source file; documentation prose about them is unaffected.
OBSOLETE_PACKAGES = ("noisyvis.visualization", "noisyvis.plotting")

REGISTRY = {'alias_names_present': ['PlotparetoFrontSubplotsHighlighted',
                         'plotMoveDeltaHistograms',
                         'plotMovementCorrelation',
                         'plotObjectiveVsDecisionScatter',
                         'plotParetoFront',
                         'plotParetoFrontAnimation',
                         'plotParetoFrontIGDVsDist',
                         'plotParetoFrontIndVsDist',
                         'plotParetoFrontNoisy',
                         'plotParetoFrontSubplots',
                         'plotParetoFrontSubplotsMulti',
                         'plotProgressPerMovementRatio'],
 'aliases': {'PlotparetoFrontSubplotsHighlighted': True,
             'plotMoveDeltaHistograms': True,
             'plotMovementCorrelation': True,
             'plotObjectiveVsDecisionScatter': True,
             'plotParetoFront': True,
             'plotParetoFrontAnimation': True,
             'plotParetoFrontIGDVsDist': True,
             'plotParetoFrontIndVsDist': True,
             'plotParetoFrontNoisy': True,
             'plotParetoFrontSubplots': True,
             'plotParetoFrontSubplotsMulti': True,
             'plotProgressPerMovementRatio': True},
 'dropdown_values': ['Basic',
                     'Subplots',
                     'SubplotsMulti',
                     'SubplotsHighlight',
                     'paretoAnimation',
                     'Noisy',
                     'IndVsDist',
                     'IGDVsDist',
                     'PPM',
                     'MoveCorr',
                     'Hist',
                     'Scatter'],
 'monolith_present': True,
 'pareto': {'Basic': {'ast': '2f68804e7368b1a3250c7bf5a53c547b928fdad4b104472a7d3472406f290e10',
                      'lookup_is_same': True,
                      'name': 'plot_basic'},
            'Hist': {'ast': '0880d2471e94dec484c1e7f2fba320e599de0a2039e07921ab221e8ea8f78d49',
                     'lookup_is_same': True,
                     'name': 'plot_move_delta_histograms'},
            'IGDVsDist': {'ast': '97a3c69fd0f4348af42abad1d286584ff01d81a0fb0628586cdc6c435894dfd1',
                          'lookup_is_same': True,
                          'name': 'plot_igd_vs_dist'},
            'IndVsDist': {'ast': '0b492122dd7e138c9389e5e78eecdad1d1cb836d14ed3aadf7dfabaf56a89fcf',
                          'lookup_is_same': True,
                          'name': 'plot_ind_vs_dist'},
            'MoveCorr': {'ast': '15796867a31fab3b3acd19f98009ecd5e0adc9a90b13bf112c73c3f0592d420c',
                         'lookup_is_same': True,
                         'name': 'plot_movement_correlation'},
            'Noisy': {'ast': 'fce09008551c562f71973f06947c47f5a52d7c628b9d0ba87dea523b1ebcd76a',
                      'lookup_is_same': True,
                      'name': 'plot_noisy'},
            'PPM': {'ast': 'bd36bce52e3806bc2eab5ad84ad9bf6f1e6eb422bbc030e308d1a82ba04f6d9e',
                    'lookup_is_same': True,
                    'name': 'plot_progress_per_movement'},
            'Scatter': {'ast': 'db728dca1f2b25eb1d14ed2da821acb7abfe954e566d621129695c3fa8a83329',
                        'lookup_is_same': True,
                        'name': 'plot_objective_vs_decision'},
            'Subplots': {'ast': 'fee65baba41485419e95a807fe8e2c0b27caad653b0b8c1ce3bf7f4510fbabbc',
                         'lookup_is_same': True,
                         'name': 'plot_subplots'},
            'SubplotsHighlight': {'ast': '1b37c149ac22b3ce11b491958346ea65d37957332de7be419c5383fdf34c7470',
                                  'lookup_is_same': True,
                                  'name': 'plot_subplots_highlighted'},
            'SubplotsMulti': {'ast': '17654de1ae41434200b5ccf99d32d2316533b6f3eef02eafe51b4d7a4e71bd5c',
                              'lookup_is_same': True,
                              'name': 'plot_subplots_multi'},
            'paretoAnimation': {'ast': '0b6e4b7120591c2df0bb0c51b0564cc953de317342eaa9d30c6cfcf8341de189',
                                'lookup_is_same': True,
                                'name': 'plot_animation'}},
 'pareto_keys': ['Basic',
                 'Subplots',
                 'SubplotsMulti',
                 'SubplotsHighlight',
                 'paretoAnimation',
                 'Noisy',
                 'IndVsDist',
                 'IGDVsDist',
                 'PPM',
                 'MoveCorr',
                 'Hist',
                 'Scatter'],
 'performance': {'box': {'ast': '404cad1d7cb32fd7003e9806b1b5306b55107e61c6d91d2d216b0c6403ebab83',
                         'lookup_is_same': True,
                         'name': 'plot_box'},
                 'box_mo': {'ast': '0455279f7a4c88b5b0bd3b95293ad16b9f4d321473706414d8f00720f7779dee',
                            'lookup_is_same': True,
                            'name': 'plot_box_mo'},
                 'line': {'ast': 'cafd6966fcb48402fa63c19ab77c6a0c2ac9979e4ec3541a306cfd6493c68ef0',
                          'lookup_is_same': True,
                          'name': 'plot_line'},
                 'line_mo': {'ast': 'd244a841dbe682f342bb92294ac1a234096e16992270a1b6f86f16554a1c0632',
                             'lookup_is_same': True,
                             'name': 'plot_line_mo'}},
 'performance_aliases': {'plot2d_box': True,
                         'plot2d_box_advanced_misjudgements_so': True,
                         'plot2d_box_evals': True,
                         'plot2d_box_misjudgements_so': True,
                         'plot2d_box_mo': True,
                         'plot2d_box_penalty': True,
                         'plot2d_line': True,
                         'plot2d_line_evals': True,
                         'plot2d_line_mo': True},
 'performance_keys': ['line', 'box', 'line_mo', 'box_mo'],
 'unknown_key_is_none': True}

PERFORMANCE_FIGURES = {'missing_columns|plot_box_advanced_misjudgements_so': '213f18620ff29dbc33424fc41d25bdec6ae533fd7a19ae7c7a488bb169cfe5db',
 'missing_columns|plot_box_evals': {'error': 'KeyError', 'message': '"[\'n_evals\'] not in index"'},
 'missing_columns|plot_box_misjudgements_so': '213f18620ff29dbc33424fc41d25bdec6ae533fd7a19ae7c7a488bb169cfe5db',
 'missing_columns|plot_box_mo': '958b619755fe13ad45905896a46c65992b3d0eb8e5ade1556668e96834ecd9a4',
 'missing_columns|plot_box_penalty': '1d6cb7f6f00cd2a4996845b6b5fb002ad438da2943334e821f15ad9e744aae49',
 'missing_columns|plot_line_mo': '958b619755fe13ad45905896a46c65992b3d0eb8e5ade1556668e96834ecd9a4',
 'plot_box_advanced_misjudgements_so|{"algo_name": "MuPlusLambda"}': '60f18fe8691114589031563e531d68fcae524603f6346a12974da3f036888cd7',
 'plot_box_evals|{"fitness_mode": "final"}': '1f4584d07eab3adde0becb7afb83093a171cdc96127ea3e3adba7f5f066f7a3d',
 'plot_box_misjudgements_so|{}': 'da2d37c524f6ecf883eff8d686ed7ee92ee7a7b36d7c94358171e6d8454b0ebf',
 'plot_box_mo|{}': '0798a209a3843a2ab01757beac75e79c35b538bafb7b1b22e3049e530e01d5af',
 'plot_box_penalty|{"fitness_mode": "best", "problem_goal": "maximise"}': '2ea9b15bdaf222dd4e1e90cd60b8540f41cbefdc7deffc0d3ae8f1e5037aee0e',
 'plot_box|{"fitness_mode": "best", "problem_goal": "maximise"}': '87708a9fc04333fc5df9e1d7458c2003af92d078d812e6d1fc37269519951b86',
 'plot_box|{"fitness_mode": "final_noisy", "problem_goal": "minimise"}': 'e9d4f73cdaf461131b9b89cb6d54883925e4b5ba4e56aaa3b2d44013ec7f7c05',
 'plot_line_evals|{"fitness_mode": "best", "show_std": false}': '75cc4076ddd51815125071bada48fdf561f8c13d733b39e2303e9ad7349f7873',
 'plot_line_evals|{"fitness_mode": "final", "show_std": true}': 'd35956cd8b1f5d38293539a75ef9f665591ef3fd677910097cc89caa11be05fd',
 'plot_line_mo|{}': 'bc0effd2841bfc8c392c82f009073442f73a6771369a26bd14a642fb04a48657',
 'plot_line|{"fitness_mode": "best", "problem_goal": "maximise"}': 'd72c710700defd887d086bea948455dd6da691c4bc224e524773c74c6c3f819c',
 'plot_line|{"fitness_mode": "best_noisy", "problem_goal": "maximise", "xaxis_title": "sigma"}': '8e49a1e33549ff742f42dbc551a9309d82cf80935ebb144c0e2caf77f40bd6ab',
 'plot_line|{"fitness_mode": "final", "problem_goal": "minimise"}': 'd871f5bc0dee160238312002701e217873b60251d815aabb7fc8d23a24bab351'}

LON_STATS = {'correlation_pairs': {'error|abs_error': {'dict': [['pearson', {'float': '0x1.742d1d96e17fbp-1'}],
                                                    ['spearman', {'float': '0x1.9999999999999p-1'}],
                                                    ['n', {'int': '4'}]]},
                       'fitness|median': {'dict': [['pearson', {'float': '0x1.fb1988ffbda82p-1'}],
                                                   ['spearman', {'float': '0x1.0000000000000p+0'}],
                                                   ['n', {'int': '4'}]]},
                       'neigh_feas|error': {'dict': [['pearson', {'float': '-0x1.b52e1ae0d712bp-3'}],
                                                     ['spearman', {'float': '0x0.0p+0'}],
                                                     ['n', {'int': '4'}]]},
                       'neigh_feas|iqr': {'dict': [['pearson', {'float': '-0x1.2f5f2f2433e06p-1'}],
                                                   ['spearman', {'float': '-0x1.9999999999999p-2'}],
                                                   ['n', {'int': '4'}]]}},
 'correlation_pair|constant': {'dict': [['pearson', None],
                                        ['spearman', {'float': '0x0.0p+0'}],
                                        ['n', {'int': '4'}]]},
 'correlation_pair|single': {'dict': [['pearson', None], ['spearman', None], ['n', {'int': '1'}]]},
 'defaults': {'AXIS_LABELS': {'dict': [['neigh_feas', 'Neighbourhood Feasibility'],
                                       ['error', 'Sampling Error'],
                                       ['abs_error', 'Absolute Error'],
                                       ['iqr', 'Sample Range (Q3-Q1)'],
                                       ['fitness', 'Fitness'],
                                       ['median', 'Median Sampled Fitness']]},
              'AXIS_OPTIONS': {'list': [{'tuple': ['neigh_feas', 'Neighbourhood Feasibility']},
                                        {'tuple': ['error', 'Sampling Error']},
                                        {'tuple': ['abs_error', 'Absolute Error']},
                                        {'tuple': ['iqr', 'Sample Range (Q3-Q1)']},
                                        {'tuple': ['fitness', 'Fitness']},
                                        {'tuple': ['median', 'Median Sampled Fitness']}]},
              'DEFAULT_PLOT_STYLE': 'scatter',
              'DEFAULT_X_AXIS': 'neigh_feas',
              'DEFAULT_Y_AXIS': 'error',
              'PLOT_STYLE_OPTIONS': {'list': [{'tuple': ['scatter', 'Scatter']},
                                              {'tuple': ['violin', 'Violin']}]}},
 'pairwise_correlations': {'list': [{'dict': [['label', 'Neighbourhood feasibility vs. error'],
                                              ['pearson', {'float': '-0x1.b52e1ae0d712bp-3'}],
                                              ['spearman', {'float': '0x0.0p+0'}],
                                              ['n', {'int': '4'}]]},
                                    {'dict': [['label', 'Neighbourhood feasibility vs. |error|'],
                                              ['pearson', {'float': '-0x1.0ad963dd9cb01p-1'}],
                                              ['spearman', {'float': '-0x1.9999999999999p-2'}],
                                              ['n', {'int': '4'}]]},
                                    {'dict': [['label', 'Neighbourhood feasibility vs. sample IQR'],
                                              ['pearson', {'float': '-0x1.2f5f2f2433e06p-1'}],
                                              ['spearman', {'float': '-0x1.9999999999999p-2'}],
                                              ['n', {'int': '4'}]]}]},
 'scatter|empty': 'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
 'scatter|error|abs_error': 'fb840c9fc6691d9ab8f71146a3cc77d210214c6a9b3b8cc91c2f27201b47f838',
 'scatter|fitness|median': 'a53a08b8a9804f5eefa85b7bc5ee483aceccc21f4fa6f0a5d3dd7fc1363b479b',
 'scatter|neigh_feas|error': '946a41b174ed6b005679a05effdabda15c2cf1b0b7daf1ec076ce1c35a676a3c',
 'scatter|neigh_feas|iqr': 'a4e9e5ace14609af2799143bcf856017da649d6a985b06f80ca48a8620c450ac',
 'stats_multi|all_empty': 'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
 'stats_multi|empty': 'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
 'stats_multi|scatter|error|abs_error': '2dee97bd31390b1d43a1a89a94e7e47a58b0fe168e40e63db17128321141e5e0',
 'stats_multi|scatter|fitness|median': 'ebe51597780fc69dea95296b05eb531c177e3fd563da05822ca72325d5d2b2b2',
 'stats_multi|scatter|neigh_feas|error': 'fe83dbf3a40230d6e4b51eb0d5ada9a5bf7da3c1516fb1d7c7d5603826846cb2',
 'stats_multi|scatter|neigh_feas|iqr': '02e1bf253fd0ca03e8bc6b2f346b3333dd672fa3b2b89e557cac5a86b56b0f5e',
 'stats_multi|violin|error|abs_error': 'f7fb7e7b26f81275eb00735be28b81b93cabf47965c861dca8b6c5133973a73d',
 'stats_multi|violin|fitness|median': '1501f042973471ec82ec76cac761f1f53571b800fa5e8ab51b9a7c392e6ad24f',
 'stats_multi|violin|neigh_feas|error': 'c182a770c9de785bdc422044fc943eee70ece54def2c6a69ddbd027b5c3d7866',
 'stats_multi|violin|neigh_feas|iqr': '5481cb9365664b33de5f4fc22c48aa868a84278d7a8bd85130598b33cd356234',
 'stats|empty': 'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
 'stats|scatter|error|abs_error': 'fb840c9fc6691d9ab8f71146a3cc77d210214c6a9b3b8cc91c2f27201b47f838',
 'stats|scatter|fitness|median': 'a53a08b8a9804f5eefa85b7bc5ee483aceccc21f4fa6f0a5d3dd7fc1363b479b',
 'stats|scatter|neigh_feas|error': '946a41b174ed6b005679a05effdabda15c2cf1b0b7daf1ec076ce1c35a676a3c',
 'stats|scatter|neigh_feas|iqr': 'a4e9e5ace14609af2799143bcf856017da649d6a985b06f80ca48a8620c450ac',
 'stats|violin|error|abs_error': 'e906685575f6204478e92754fcb98a1722bcfee0dfada91ac7fedc7f927b7ffc',
 'stats|violin|fitness|median': 'e08359267cbebc63241edd56c370fd22a41d3774dfd997f8e6b31a6f148e92ae',
 'stats|violin|neigh_feas|error': '43801d4a26345925b25b00e42e3e68126fc23c0cc7e96daa4c607bc60ef400f5',
 'stats|violin|neigh_feas|iqr': '0e3b292bcfd0abc3adc7df55e547ebda136fceac43ec834ecbef2285a35960f0',
 'table|correlations': 'be60b2984201682baa106044d08adcf9f5bf1b2311712ee929c382b0415a69f1',
 'table|correlations_empty': 'ecb2d5b823fb6bb39b58a7257df8b4e55d4d2b22c81a9c80162f5baf1250ce8d',
 'table|selected': '37d7174e94afc772d2d0c327b4559ac00e7c7fee3bd51582a6bcacc9ff47e3c5',
 'table|selected_none': 'c54f5d424c56eba65be95e1b1adcc5db85592c5dd100d6ce7532430232c0080a',
 'violin|empty': 'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
 'violin|error|abs_error': 'e906685575f6204478e92754fcb98a1722bcfee0dfada91ac7fedc7f927b7ffc',
 'violin|fitness|median': 'e08359267cbebc63241edd56c370fd22a41d3774dfd997f8e6b31a6f148e92ae',
 'violin|neigh_feas|error': '43801d4a26345925b25b00e42e3e68126fc23c0cc7e96daa4c607bc60ef400f5',
 'violin|neigh_feas|iqr': '0e3b292bcfd0abc3adc7df55e547ebda136fceac43ec834ecbef2285a35960f0'}

BODIES = {'dashboard/Dashboard.py': {'ast': '0d5341c4f0d00aeaf3e0425dfa682d40af43e5b7a33385b4a744e79c5b735bc9',
                            'literals': '132f86940a8cf3f9891b2d3ab6fd25514e312cec6ee3e635e313ea7ec0467e58',
                            'statements': 75},
 'dashboard/DashboardHelpers.py': {'ast': 'a2f8637b5bc9b75f30cdfbe3954a31adffac64281b8bbcc355ebf87af53fa714',
                                   'literals': 'f5426899c41d6be7373fb1cd4227501d84bbe098001d77bc81064b2a55e73e09',
                                   'statements': 11},
 'dashboard/layout/components.py': {'ast': 'd2bf964945b3e2d88f7cc49b5e479cc20a912b5b07c242efe4afd93af4019dbe',
                                    'literals': 'f3a655987d7c1a1706f822d1704d2c38709e8ada9502be55bcb08ade1e1c6e4f',
                                    'statements': 24}}

IMPORTS = {'dashboard/Dashboard.py': {'modules': ['noisyvis.plotting',
                                        'noisyvis.plotting.performance',
                                        'noisyvis.visualization'],
                            'name_count': 39,
                            'names': ['LON_SCATTER_AXIS_LABELS',
                                      'LON_SCATTER_DEFAULT_PLOT_STYLE',
                                      'LON_SCATTER_DEFAULT_X_AXIS',
                                      'LON_SCATTER_DEFAULT_Y_AXIS',
                                      'PlotConfig',
                                      'add_lon_edges',
                                      'add_lon_nodes',
                                      'add_mo_fronts',
                                      'add_prior_noise_stn_algo_pov',
                                      'add_prior_noise_stn_v4',
                                      'add_prior_noise_stn_v5',
                                      'add_stn_trajectories',
                                      'build_all_traces',
                                      'build_correlation_table',
                                      'build_selected_correlation_display',
                                      'calculate_lon_statistics',
                                      'calculate_positions',
                                      'compute_correlation_pair',
                                      'compute_node_feasibility_error',
                                      'compute_pairwise_correlations',
                                      'create_axis_settings',
                                      'create_figure',
                                      'create_guide_traces',
                                      'debug_mo_counts',
                                      'generate_run_summary_string',
                                      'get_pareto_plot',
                                      'parse_callback_inputs',
                                      'plot2d_box',
                                      'plot2d_box_advanced_misjudgements_so',
                                      'plot2d_box_evals',
                                      'plot2d_box_misjudgements_so',
                                      'plot2d_box_mo',
                                      'plot2d_box_penalty',
                                      'plot2d_line',
                                      'plot2d_line_evals',
                                      'plot2d_line_mo',
                                      'plot_lon_stats',
                                      'plot_lon_stats_multi',
                                      'style_nodes'],
                            'resolved': {'LON_SCATTER_AXIS_LABELS': {'canon': {'dict': [['neigh_feas',
                                                                                         'Neighbourhood '
                                                                                         'Feasibility'],
                                                                                        ['error',
                                                                                         'Sampling Error'],
                                                                                        ['abs_error',
                                                                                         'Absolute Error'],
                                                                                        ['iqr',
                                                                                         'Sample Range '
                                                                                         '(Q3-Q1)'],
                                                                                        ['fitness',
                                                                                         'Fitness'],
                                                                                        ['median',
                                                                                         'Median Sampled '
                                                                                         'Fitness']]},
                                                                     'kind': 'value'},
                                         'LON_SCATTER_DEFAULT_PLOT_STYLE': {'canon': 'scatter',
                                                                            'kind': 'value'},
                                         'LON_SCATTER_DEFAULT_X_AXIS': {'canon': 'neigh_feas',
                                                                        'kind': 'value'},
                                         'LON_SCATTER_DEFAULT_Y_AXIS': {'canon': 'error', 'kind': 'value'},
                                         'PlotConfig': {'ast': '6508fdf1b8c1a8380eb6b51862dc52a0b873a50202e01e9bac4c76dde75bf903',
                                                        'kind': 'class',
                                                        'name': 'PlotConfig'},
                                         'add_lon_edges': {'ast': '5c6f73c6f71997c13e9b1452614e546adb34cf291956e851d25b1963ddd040d1',
                                                           'kind': 'callable',
                                                           'name': 'add_lon_edges'},
                                         'add_lon_nodes': {'ast': '16374f54f260ab1e2fee43f985834ce62309305b8cc8fbcd9fde6a709a5a1eb5',
                                                           'kind': 'callable',
                                                           'name': 'add_lon_nodes'},
                                         'add_mo_fronts': {'ast': 'd7d09e7be956b3ab28c5c69c410b4d3c508d8916c6dd009e84730d0953f9d788',
                                                           'kind': 'callable',
                                                           'name': 'add_mo_fronts'},
                                         'add_prior_noise_stn_algo_pov': {'ast': '62496cb4794186ccf7b1fbd987e708bd40e2fa8fd2bfab25f0871667c0befcf8',
                                                                          'kind': 'callable',
                                                                          'name': 'add_prior_noise_stn_algo_pov'},
                                         'add_prior_noise_stn_v4': {'ast': 'c42e2f94f099b62623e1462a2b54f6fc956db9b4f5536412afac9556ec1486d6',
                                                                    'kind': 'callable',
                                                                    'name': 'add_prior_noise_stn_v4'},
                                         'add_prior_noise_stn_v5': {'ast': '4fd049f24ea1e06140d2f54103162df757a9b10482244b7cfa3997729104a01a',
                                                                    'kind': 'callable',
                                                                    'name': 'add_prior_noise_stn_v5'},
                                         'add_stn_trajectories': {'ast': '18818678708be3d6a31954bf43fadd7d453843d32a390c773116911fcd9c318b',
                                                                  'kind': 'callable',
                                                                  'name': 'add_stn_trajectories'},
                                         'build_all_traces': {'ast': 'bbf7a780d64370f7afd256b5936c88108af496985a4fd6621068f61359bff168',
                                                              'kind': 'callable',
                                                              'name': 'build_all_traces'},
                                         'build_correlation_table': {'ast': 'a470beefdd4ef2dba56e076be9b3c67db8df1b0f9071bac568d6a3fc657090f5',
                                                                     'kind': 'callable',
                                                                     'name': 'build_correlation_table'},
                                         'build_selected_correlation_display': {'ast': '48fbfa39552765bff9d0c3d72e9c4019914853101e68add1ce530f40f2308cb3',
                                                                                'kind': 'callable',
                                                                                'name': 'build_selected_correlation_display'},
                                         'calculate_lon_statistics': {'ast': '6560bdcd253d53dac15d3376894ba6c60d0049f0bed6b4c495ae69f7e4b04af0',
                                                                      'kind': 'callable',
                                                                      'name': 'calculate_lon_statistics'},
                                         'calculate_positions': {'ast': '860cd1cb5ad743ff17302e125b20070fcae3363c6b31ca0d4090f42325a55de3',
                                                                 'kind': 'callable',
                                                                 'name': 'calculate_positions'},
                                         'compute_correlation_pair': {'ast': 'f1f64f65296abc44db293a60823598354ac8709865b8049cf503e0297de26546',
                                                                      'kind': 'callable',
                                                                      'name': 'compute_correlation_pair'},
                                         'compute_node_feasibility_error': {'ast': '33f16584712f6f4cc8657512d9f97c22e28c68dd7fc3eaa46af9efc24a22f7d2',
                                                                            'kind': 'callable',
                                                                            'name': 'compute_node_feasibility_error'},
                                         'compute_pairwise_correlations': {'ast': '289a9f3d3dc61a22f85e907cea87064c94623b22acc3ad08cd4dc8dfd5657501',
                                                                           'kind': 'callable',
                                                                           'name': 'compute_pairwise_correlations'},
                                         'create_axis_settings': {'ast': '7db8637c06d7e280bb577779ddab3db65160061a748bc419160fde3176eeea83',
                                                                  'kind': 'callable',
                                                                  'name': 'create_axis_settings'},
                                         'create_figure': {'ast': 'b9f77986cfda392064269e7e1c1cbce066c4fc02dd904bb0ba5f037f5e46154b',
                                                           'kind': 'callable',
                                                           'name': 'create_figure'},
                                         'create_guide_traces': {'ast': '4b601205206e5b59e9d278af57be55e15271acede44ecd99e61144fecf06b8b8',
                                                                 'kind': 'callable',
                                                                 'name': 'create_guide_traces'},
                                         'debug_mo_counts': {'ast': 'c463074b4ea8f62d2554f140c1bcc73a9bc26978d1e09a864f69746a73ec976c',
                                                             'kind': 'callable',
                                                             'name': 'debug_mo_counts'},
                                         'generate_run_summary_string': {'ast': '6c7689af454a79cd0988ba32b2bd5c4fa524ef38ea318fe60edc6405de0b4f29',
                                                                         'kind': 'callable',
                                                                         'name': 'generate_run_summary_string'},
                                         'get_pareto_plot': {'ast': '8a660705f72a757524625733cf214685af2dbe09d7c2bf1d58825c804ef4b1fc',
                                                             'kind': 'callable',
                                                             'name': 'get_pareto_plot'},
                                         'parse_callback_inputs': {'ast': 'b3211930c9b4e6d78f5c6ef5b39b6793911b31be5b3150fcb3f1e7e6d533d899',
                                                                   'kind': 'callable',
                                                                   'name': 'parse_callback_inputs'},
                                         'plot2d_box': {'ast': '404cad1d7cb32fd7003e9806b1b5306b55107e61c6d91d2d216b0c6403ebab83',
                                                        'kind': 'callable',
                                                        'name': 'plot_box'},
                                         'plot2d_box_advanced_misjudgements_so': {'ast': 'ee1f1585217b35d9553d4ef450ebde063fef616af2f551a8e3704d82298744ea',
                                                                                  'kind': 'callable',
                                                                                  'name': 'plot_box_advanced_misjudgements_so'},
                                         'plot2d_box_evals': {'ast': '7f72d7aadff43076dde88caf850035a44a4c705e413d47e7dc6e816e1aa2ca64',
                                                              'kind': 'callable',
                                                              'name': 'plot_box_evals'},
                                         'plot2d_box_misjudgements_so': {'ast': 'fc101959c79d3912dd47c02ab8acddf501817187ae6c4f3c686f19d5adb9c454',
                                                                         'kind': 'callable',
                                                                         'name': 'plot_box_misjudgements_so'},
                                         'plot2d_box_mo': {'ast': '0455279f7a4c88b5b0bd3b95293ad16b9f4d321473706414d8f00720f7779dee',
                                                           'kind': 'callable',
                                                           'name': 'plot_box_mo'},
                                         'plot2d_box_penalty': {'ast': 'bf3d58160ec51aa00f33134129d525d0d6a967886430b31f8964547e95834357',
                                                                'kind': 'callable',
                                                                'name': 'plot_box_penalty'},
                                         'plot2d_line': {'ast': 'cafd6966fcb48402fa63c19ab77c6a0c2ac9979e4ec3541a306cfd6493c68ef0',
                                                         'kind': 'callable',
                                                         'name': 'plot_line'},
                                         'plot2d_line_evals': {'ast': 'eb0485a5ef0edbe43c3361dd5af2a3a71d9cb517217be904fe6f349791ff4158',
                                                               'kind': 'callable',
                                                               'name': 'plot_line_evals'},
                                         'plot2d_line_mo': {'ast': 'd244a841dbe682f342bb92294ac1a234096e16992270a1b6f86f16554a1c0632',
                                                            'kind': 'callable',
                                                            'name': 'plot_line_mo'},
                                         'plot_lon_stats': {'ast': '3cf45315991737c2e0bd23c2cd733afa000a14add3f4cff505279f49cb749dc8',
                                                            'kind': 'callable',
                                                            'name': 'plot_lon_stats'},
                                         'plot_lon_stats_multi': {'ast': '272fbb0b7a5c90b3a7d2c47f04ce75d02f78cc7ff509abd6a942579048040815',
                                                                  'kind': 'callable',
                                                                  'name': 'plot_lon_stats_multi'},
                                         'style_nodes': {'ast': 'c5c2d640448f6ab7dd351810746c9c323f54cd8182a301433bf482903152f02b',
                                                         'kind': 'callable',
                                                         'name': 'style_nodes'}}},
 'dashboard/DashboardHelpers.py': {'modules': ['noisyvis.plotting.performance'],
                                   'name_count': 4,
                                   'names': ['plot2d_box',
                                             'plot2d_box_mo',
                                             'plot2d_line',
                                             'plot2d_line_mo'],
                                   'resolved': {'plot2d_box': {'ast': '404cad1d7cb32fd7003e9806b1b5306b55107e61c6d91d2d216b0c6403ebab83',
                                                               'kind': 'callable',
                                                               'name': 'plot_box'},
                                                'plot2d_box_mo': {'ast': '0455279f7a4c88b5b0bd3b95293ad16b9f4d321473706414d8f00720f7779dee',
                                                                  'kind': 'callable',
                                                                  'name': 'plot_box_mo'},
                                                'plot2d_line': {'ast': 'cafd6966fcb48402fa63c19ab77c6a0c2ac9979e4ec3541a306cfd6493c68ef0',
                                                                'kind': 'callable',
                                                                'name': 'plot_line'},
                                                'plot2d_line_mo': {'ast': 'd244a841dbe682f342bb92294ac1a234096e16992270a1b6f86f16554a1c0632',
                                                                   'kind': 'callable',
                                                                   'name': 'plot_line_mo'}}},
 'dashboard/layout/components.py': {'modules': ['noisyvis.visualization'],
                                    'name_count': 5,
                                    'names': ['LON_SCATTER_AXIS_OPTIONS',
                                              'LON_SCATTER_DEFAULT_PLOT_STYLE',
                                              'LON_SCATTER_DEFAULT_X_AXIS',
                                              'LON_SCATTER_DEFAULT_Y_AXIS',
                                              'LON_SCATTER_PLOT_STYLE_OPTIONS'],
                                    'resolved': {'LON_SCATTER_AXIS_OPTIONS': {'canon': {'list': [{'tuple': ['neigh_feas',
                                                                                                            'Neighbourhood '
                                                                                                            'Feasibility']},
                                                                                                 {'tuple': ['error',
                                                                                                            'Sampling '
                                                                                                            'Error']},
                                                                                                 {'tuple': ['abs_error',
                                                                                                            'Absolute '
                                                                                                            'Error']},
                                                                                                 {'tuple': ['iqr',
                                                                                                            'Sample '
                                                                                                            'Range '
                                                                                                            '(Q3-Q1)']},
                                                                                                 {'tuple': ['fitness',
                                                                                                            'Fitness']},
                                                                                                 {'tuple': ['median',
                                                                                                            'Median '
                                                                                                            'Sampled '
                                                                                                            'Fitness']}]},
                                                                              'kind': 'value'},
                                                 'LON_SCATTER_DEFAULT_PLOT_STYLE': {'canon': 'scatter',
                                                                                    'kind': 'value'},
                                                 'LON_SCATTER_DEFAULT_X_AXIS': {'canon': 'neigh_feas',
                                                                                'kind': 'value'},
                                                 'LON_SCATTER_DEFAULT_Y_AXIS': {'canon': 'error',
                                                                                'kind': 'value'},
                                                 'LON_SCATTER_PLOT_STYLE_OPTIONS': {'canon': {'list': [{'tuple': ['scatter',
                                                                                                                  'Scatter']},
                                                                                                       {'tuple': ['violin',
                                                                                                                  'Violin']}]},
                                                                                    'kind': 'value'}}}}

PARETO_FIGURES = {'Basic|{}': 'dcfa2f98b9f812460ba0df35f383508ec3392d74d83e59cea836942961e8c4ac',
 'Hist|{"IndVsDist_IndType": "CleanHV"}': 'f634401b156e846ef98df58ecfa5c3e012058014ac2711a709321831c31947ee',
 'Hist|{"IndVsDist_IndType": "IGD"}': 'f634401b156e846ef98df58ecfa5c3e012058014ac2711a709321831c31947ee',
 'Hist|{"IndVsDist_IndType": "NoisyHV"}': '453788c843a7b20bb0b48945f390cf139d206447f374a73139c038b3f9dcef9f',
 'IGDVsDist|{"distance_method": "cumulative", "nruns": 1}': 'f775c9cbcec4f429a0adcde9210dd5770b7280b64e9f7784b7364101a8294c86',
 'IGDVsDist|{"distance_method": "isomap", "nruns": 1}': 'dc9ecbfef015053ecf4fd80dd97736d3d2837202c176499d03f66a1d49a0581f',
 'IGDVsDist|{"distance_method": "mds", "nruns": 1}': '991beaee895ae166b42b58636cab55c82ac62c23e587c814ff9b5f908baa54e9',
 'IGDVsDist|{"distance_method": "raw", "nruns": 1}': '293eb90986a6631bcae7077a42e2270ed31ea87d6eeeafc1dfa3d8fc08119109',
 'IGDVsDist|{"distance_method": "tsne", "nruns": 1}': '181da8433881a64939f65545a2d5de673fe383da2d6cc126cf746f61c90a5bd0',
 'IndVsDist|{"distance_method": "cumulative", "nruns": 1}': 'bbf6192b754ac6e92158dfde1870db6d8c81f71194b8bbcc7e4f9dccac1f06b7',
 'IndVsDist|{"distance_method": "isomap", "nruns": 1}': '627922079855afd8fd4fbb6b5e691c7344ea8695d7f00b6ed4d60750b609d960',
 'IndVsDist|{"distance_method": "mds", "nruns": 1}': '035fa11297697bd0fd9a035d3e3ee803850a80f82d8f767785e982289d9d7793',
 'IndVsDist|{"distance_method": "raw", "nruns": 1}': '1d26e1a3f6281392bbdaa846a0ac71ae0b2c49a6dade871c10310f7d91f62e63',
 'IndVsDist|{"distance_method": "tsne", "nruns": 1}': 'a0dbce819493083b08250e1d651bab6c191e1026deafc14b23124d765176ca00',
 'MoveCorr|{"IndVsDist_IndType": "CleanHV", "window": 5}': '63334c1b26881667d464d17e8ae10eab20aeff26085761b0bfa27e6e6d056124',
 'MoveCorr|{"IndVsDist_IndType": "IGD", "window": 5}': '63334c1b26881667d464d17e8ae10eab20aeff26085761b0bfa27e6e6d056124',
 'MoveCorr|{"IndVsDist_IndType": "NoisyHV", "window": 5}': '2c3d4b24321cc6a2d09ae680fb43ca520d1331001d664c2cabdc8ca16c375c77',
 'Noisy|{}': '463dd0eefd9b178a5be7fdd85c66a16cd5346f8a2f7f41b423c817b884e807fe',
 'PPM|{}': 'bb609ad1b1d15164d7f450fb1146d7210682f83125d11f334e5195772ab41fef',
 'Scatter|{"IndVsDist_IndType": "CleanHV"}': 'f3483da60e31c6f642d64974079df9cdb334f812e2ec3c4ae55678788f581d5e',
 'Scatter|{"IndVsDist_IndType": "IGD"}': 'f3483da60e31c6f642d64974079df9cdb334f812e2ec3c4ae55678788f581d5e',
 'Scatter|{"IndVsDist_IndType": "NoisyHV"}': '80b47a509fdebb971a44fbdabc976a07ae3b7ea1f91bbe36ac96e9afc90dcf6d',
 'SubplotsHighlight|{}': '617a15df2a002be50fb5a117808e653a8ce7d5aa01cbebd3baa841f84f22dfa8',
 'SubplotsMulti|{"nruns": 2}': 'babaeaf0a7d9c95d1e75cfb78d551b43ba558c85e35db322856cec0771ffdf0f',
 'Subplots|{}': '029032bf88ed864da08f9d59bb59ff905214d0ed6f09634de4bc514d0a308a91',
 'create_pareto_plot|unknown': '140860e83409afc1d2f9d3e28221aa2e7d000c32264e96344976825eed10e540',
 'empty|Basic': '6323ba24c222434855da4c7a5a3e2a6e81b56e581c8c2434eaf8347185df2083',
 'empty|Hist': '4a74584c713b3939fa71d7123f3caba381282f306c2230b839e88e5bb60c9a10',
 'empty|IGDVsDist': '6323ba24c222434855da4c7a5a3e2a6e81b56e581c8c2434eaf8347185df2083',
 'empty|IndVsDist': '6323ba24c222434855da4c7a5a3e2a6e81b56e581c8c2434eaf8347185df2083',
 'empty|MoveCorr': '0213c2468810ec0278e4e5cf555e968f1b5a6af1bd133d5e5ac0630d6ced08dc',
 'empty|Noisy': '6323ba24c222434855da4c7a5a3e2a6e81b56e581c8c2434eaf8347185df2083',
 'empty|PPM': 'c94f9f1a5c13f39179523ccf08e37ee2e35b8c75930b14c3327c5e6d06962f3d',
 'empty|Scatter': '4705318c81849d25fb3695b7e4ec1ac6efa2c7d4eb0841af28d649e1a9ca32f3',
 'empty|Subplots': '460029fa309882f07ec50a4a687eb3ea3162b9ee9d74a525c21bb83dff37113f',
 'empty|SubplotsHighlight': '460029fa309882f07ec50a4a687eb3ea3162b9ee9d74a525c21bb83dff37113f',
 'empty|SubplotsMulti': '460029fa309882f07ec50a4a687eb3ea3162b9ee9d74a525c21bb83dff37113f',
 'empty|paretoAnimation': '460029fa309882f07ec50a4a687eb3ea3162b9ee9d74a525c21bb83dff37113f',
 'paretoAnimation|{}': '0e04f11b63fa877bbc50b9b5ab6c200b4b96d147528fed33139ee142ac0c0768'}

MIXED = {'M1': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'guide_series',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'size',
                                        'solution',
                                        'start_node',
                                        'step',
                                        'type',
                                        'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'evals', 'norm_weight', 'weight'],
                          'edge_count': 182,
                          'edge_types': ['LON', 'Noise', 'NoisyPath_SO', 'STN', 'STN_ALT'],
                          'edges': '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                          'graph_attrs': {'dict': []},
                          'node_count': 161,
                          'node_ids': 'fa1de3c1a730822c4d93f1ef1fec9f828aedb5d2f5cb763a3178e760ae3ce113',
                          'nodes': '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                          'types': ['LON', 'None', 'STN', 'STN_ALT', 'guide']},
        'calls': ['parse_callback_inputs',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  '_add_guide_nodes',
                  'calculate_positions',
                  'build_all_traces',
                  'create_guide_traces',
                  'create_axis_settings',
                  'create_figure',
                  'add_lon_nodes',
                  'compute_node_feasibility_error',
                  'add_lon_nodes',
                  'compute_node_feasibility_error',
                  'compute_node_feasibility_error',
                  'plot_lon_stats_multi',
                  'compute_correlation_pair',
                  'build_selected_correlation_display',
                  'compute_pairwise_correlations',
                  'build_correlation_table'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': '0297a1da37ad89f87e31dbdc8126e3be22d2f9e32a53dfbc15a0daeb2c9d0d12',
        'late_lon_population': [16, 18],
        'output_types': ['Figure', 'Div', 'DataTable', 'DataTable', 'Figure', 'DataTable', 'DataTable'],
        'outputs': ['0297a1da37ad89f87e31dbdc8126e3be22d2f9e32a53dfbc15a0daeb2c9d0d12',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    '78f6f0215f940c768f24cda1bbcbc66651d9927ca415ecf929d1929c5b7c8b5b',
                    '7448027897cb71a1ae5da99f069f5beaefab65632c7b97316f49082df63ee189',
                    '9050897355e7bfbbb3e0a200e4b21348b4c2443988c310cd84e2c8a5c322980b'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 161,
                      'dims': [2],
                      'items': '5004c269392e983dac057b7ad8de3cb9c7b1ef4fb09b1fb69c635ecf25091474',
                      'keys': 'fa1de3c1a730822c4d93f1ef1fec9f828aedb5d2f5cb763a3178e760ae3ce113'},
        'results': [['add_stn_trajectories',
                     'aa02a7a7ff9ccd1933989923b4eb0e52b8179ec92ec48c68fecb571321d64f71'],
                    ['add_stn_trajectories',
                     'e68ae1bd1feb4ff4c288bc5274d4c1f9a04bef8b71c974e8c0860dda121e6a84'],
                    ['add_lon_nodes', 'e53d0cda3a49fb47c4af4e627c76ec30dc11c663bc3d9e3613629c4e935ed0a4'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27'],
                    ['add_lon_nodes', '40b1c20af27a3e8764616c8f86e63ed540e906418816f8da61a1fcc779af8124'],
                    ['compute_node_feasibility_error',
                     'ebf1f54a049df6bf9d043dd7a63f0c00fe904f70d0064948cc1cab355cec3bcd'],
                    ['add_lon_nodes', '98eccbe5a7616217dcdc2edf4d37731bd36a089e0420676d6092906f2f047d1b'],
                    ['compute_node_feasibility_error',
                     '745a1af93c6e4eb03d7fced17ebbdc201514578056530a107801006a9940a433'],
                    ['compute_node_feasibility_error',
                     'a65e6b80659fd91bc200560249e2b02a0118adb97ceed6c3af78478e1bc99513'],
                    ['compute_correlation_pair',
                     '3e4d14258a70ac7056632569e23d4f6236e6f7900c65bf36a666c7411ddd20e1'],
                    ['compute_pairwise_correlations',
                     '98ea4ccf070634dbf04ea36f060c9b3a0cc259929cba71ac947891b91884dd2d']],
        'stdout': '5d0ba1842e1c8ef9f9813a5713870413c2b1c16c0480236546b0ddb247c4daa6',
        'stdout_lines': 31,
        'steps': [['add_stn_trajectories',
                   '785851c8b0e7ba580f8fed858d7f9c6cbfbd81f43888d33f53073abd9a6b5fec',
                   '6ca03da15d9bd842df8ad1fbd57879b7a1024763fd71a4af3d0fafc4830848fa',
                   62],
                  ['add_stn_trajectories',
                   'a01adabccf46e2eb654a2908fe0c8615963398dc4dd1210fbfbb6be37f2fbc0d',
                   '5507c1b5624c90aa3fdcb4a1b3b43f70c73e90effc554cd1cf8b0d25e2516248',
                   108],
                  ['debug_mo_counts',
                   'a01adabccf46e2eb654a2908fe0c8615963398dc4dd1210fbfbb6be37f2fbc0d',
                   '5507c1b5624c90aa3fdcb4a1b3b43f70c73e90effc554cd1cf8b0d25e2516248',
                   108],
                  ['add_lon_nodes',
                   '52b8026bfa538bf1b11d0d43dad888c7cd0ee1534054d555053c115d00863d6c',
                   '5507c1b5624c90aa3fdcb4a1b3b43f70c73e90effc554cd1cf8b0d25e2516248',
                   140],
                  ['add_lon_edges',
                   '52b8026bfa538bf1b11d0d43dad888c7cd0ee1534054d555053c115d00863d6c',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   140],
                  ['style_nodes',
                   '9cbe4dd33ea783036a766c1a47e2ba1f1c067683480cb8e8b73769433849b207',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   140],
                  ['calculate_lon_statistics',
                   '9cbe4dd33ea783036a766c1a47e2ba1f1c067683480cb8e8b73769433849b207',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   140],
                  ['_add_guide_nodes',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['calculate_positions',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['build_all_traces',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['create_guide_traces',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['create_axis_settings',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['add_lon_nodes',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['compute_node_feasibility_error',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['add_lon_nodes',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['compute_node_feasibility_error',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161],
                  ['compute_node_feasibility_error',
                   '0355f8583013266b8e40493819ea7639b6570efc9257f2065f8ad1e9f1db44be',
                   '88d09dd23455824fc72d07f1a3cf9e39acacaf7c426d2f7a8af446c1e7d2f23a',
                   161]]},
 'M10': {'advanced_traces': [],
         'before_layout': {'attr_keys': ['color',
                                         'count_estimated_adopted',
                                         'count_estimated_discarded',
                                         'end_node',
                                         'estimated_fitness_adopted',
                                         'estimated_fitness_discarded',
                                         'evals',
                                         'fitness',
                                         'fitness_boxplot_stats',
                                         'is_noisy',
                                         'iterations',
                                         'run_idx',
                                         'size',
                                         'solution',
                                         'start_node',
                                         'step',
                                         'type'],
                           'class': 'MultiDiGraph',
                           'edge_attr_keys': ['color', 'edge_type', 'evals', 'weight'],
                           'edge_count': 7,
                           'edge_types': ['Noise', 'NoisyPath_SO', 'STN'],
                           'edges': '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                           'graph_attrs': {'dict': []},
                           'node_count': 6,
                           'node_ids': '2c7c40c7886615431cb052d1689492e5a276541a0151089c142ce328cff61eac',
                           'nodes': 'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                           'types': ['None', 'STN']},
         'calls': ['parse_callback_inputs',
                   'add_stn_trajectories',
                   'generate_run_summary_string',
                   'debug_mo_counts',
                   'style_nodes',
                   'calculate_lon_statistics',
                   'calculate_positions',
                   'build_all_traces',
                   'create_axis_settings',
                   'create_figure',
                   'plot_lon_stats'],
         'diagnostic': False,
         'error': None,
         'figure_semantic': 'fab87fefe8484d847d34ae97111fa5a632a31f8e5a79dadbdf0fab6a608bb3ed',
         'late_lon_population': [],
         'output_types': ['Figure', 'Div', 'DataTable', 'Div', 'Figure', 'Div', 'Div'],
         'outputs': ['fab87fefe8484d847d34ae97111fa5a632a31f8e5a79dadbdf0fab6a608bb3ed',
                     '5098b937c49cb15f6eee52c2a2d16814116efdcb09fa78ef52609e407d9baffe',
                     'a4d492e63112f520e621f58fea7b404f4f3839b2c4f988fdb78b1f322b2cfd24',
                     '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                     'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                     '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                     'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
         'positions': {'all_finite': True,
                       'class': 'dict',
                       'count': 6,
                       'dims': [2],
                       'items': '160327850ec01b36fcbf3d7aa788b6ec2517b89bea931454b208cde760bc0b54',
                       'keys': '2c7c40c7886615431cb052d1689492e5a276541a0151089c142ce328cff61eac'},
         'results': [['add_stn_trajectories',
                      '6f3becfd30b48ae0391eaf76f51f1a083d26de54c79ba09407d5f0d2c797e717'],
                     ['calculate_lon_statistics',
                      '7acc563833b65e51134d754997a7c9abc54882806b587194f0559a102e4ecf6f']],
         'stdout': '5da84b8f1c3ab4a7275141c253d41585ec2c324954a55d02e8db2e16bfb88e92',
         'stdout_lines': 15,
         'steps': [['add_stn_trajectories',
                    'f01b201fe1ea6ff506233beeebe6659f86ccc7c381648587c88ffa3676603856',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['debug_mo_counts',
                    'f01b201fe1ea6ff506233beeebe6659f86ccc7c381648587c88ffa3676603856',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['style_nodes',
                    'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['calculate_lon_statistics',
                    'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['calculate_positions',
                    'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['build_all_traces',
                    'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6],
                   ['create_axis_settings',
                    'd2e8bfa79fabaaf9c87f6c63852ea29b48e480c89fe08ccd05a1d452918630c8',
                    '06aa7a97789a5419a0e651afefd4a2f01ba9e03e89e3b9009c9ac1a412873217',
                    6]]},
 'M2': {'advanced_traces': ['Advanced misjudgement: increasing noise'],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'series_idx',
                                        'size',
                                        'sol_idx',
                                        'solution',
                                        'start_node',
                                        'type',
                                        'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'evals', 'is_noisy', 'weight'],
                          'edge_count': 169,
                          'edge_types': ['LON', 'Noise_SO', 'STN_ALT', 'STN_SO'],
                          'edges': '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                          'graph_attrs': {'dict': []},
                          'node_count': 163,
                          'node_ids': '58d72784993f110e1278f25495fad6eb8505e601ee812bf306445e9daa3e74f1',
                          'nodes': 'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                          'types': ['LON', 'STN_ALT', 'STN_SO', 'STN_SO_Noise']},
        'calls': ['parse_callback_inputs',
                  'add_prior_noise_stn_v4',
                  'generate_run_summary_string',
                  'add_prior_noise_stn_v4',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'plot_lon_stats'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': '1eba7b223dc51446c9e0d0b528a5f6c9ac7e7433af48c78d7c4963d15b45440b',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'DataTable', 'DataTable', 'Figure', 'Div', 'Div'],
        'outputs': ['e8c99444127302d34af0f4f5df4106f8254776e431ebd2a9f1b78e2ee038d2b8',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 163,
                      'dims': [2],
                      'items': '53b28d2c49bb2b4ca9a9e18343ad04f13b3e3105a638c4ff8f52d8fe6b129ac5',
                      'keys': '4000e9523fd76841e30263eed783741429dbba33799334bd2eb72c60d9e72056'},
        'results': [['add_lon_nodes', '8ddf5ecc4e79211795061118723cbf7022f06958e7b94245d683de4389959dea'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27']],
        'stdout': 'afebfd1066a4d67705b52f65e0507bd4e037100b3b4e4ca42e9353fd3a115cc3',
        'stdout_lines': 3227,
        'steps': [['add_prior_noise_stn_v4',
                   '6d9e245447572574e5e8de1b9bb0ef2044485d56f57f7dcc41120416d7ecabb7',
                   '1fce2dba33d6e9f78b7d8ca474df6b51bdb2dd36a2aaf57dda15be236b0daaaf',
                   72],
                  ['add_prior_noise_stn_v4',
                   '809fb81b7e6cd4ae950c0f050b3d6bd1e7fe85a835d09980125fc416cd5e1b03',
                   'ad6c5c86eabb4d2e5e421eaa7a9e26de890e1fce4f62e17a0e64f3c3390ca6b4',
                   131],
                  ['debug_mo_counts',
                   '809fb81b7e6cd4ae950c0f050b3d6bd1e7fe85a835d09980125fc416cd5e1b03',
                   'ad6c5c86eabb4d2e5e421eaa7a9e26de890e1fce4f62e17a0e64f3c3390ca6b4',
                   131],
                  ['add_lon_nodes',
                   '16cb363fd052bdee0464f05448f3cc7941427c33e7466e4c053d56307c5114c6',
                   'ad6c5c86eabb4d2e5e421eaa7a9e26de890e1fce4f62e17a0e64f3c3390ca6b4',
                   163],
                  ['add_lon_edges',
                   '16cb363fd052bdee0464f05448f3cc7941427c33e7466e4c053d56307c5114c6',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163],
                  ['style_nodes',
                   'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163],
                  ['calculate_lon_statistics',
                   'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163],
                  ['calculate_positions',
                   'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163],
                  ['build_all_traces',
                   'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163],
                  ['create_axis_settings',
                   'ceabeba558fe355e6043f859e64ec0021a1789f8784159599a2bc1fdd1cadefe',
                   '488326d2c4224ded8e7e491e2472785b5cd3df42461eec152215cdced866d70e',
                   163]]},
 'M3': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'series_idx',
                                        'size',
                                        'sol_idx',
                                        'solution',
                                        'start_node',
                                        'type',
                                        'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color',
                                             'edge_type',
                                             'evals',
                                             'is_noisy',
                                             'norm_weight',
                                             'weight'],
                          'edge_count': 145,
                          'edge_types': ['LON', 'Noise_SO', 'NoisyPath_SO', 'STN_SO'],
                          'edges': '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                          'graph_attrs': {'dict': []},
                          'node_count': 106,
                          'node_ids': '0b8f5bf19f60a5013cbac6c4d8bec8583b9c708c949c8105de327309c3445c7a',
                          'nodes': '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                          'types': ['LON', 'STN_SO', 'STN_SO_Noise']},
        'calls': ['parse_callback_inputs',
                  'add_prior_noise_stn_v5',
                  'generate_run_summary_string',
                  'add_prior_noise_stn_v5',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'add_lon_nodes',
                  'compute_node_feasibility_error',
                  'add_lon_nodes',
                  'compute_node_feasibility_error',
                  'compute_node_feasibility_error',
                  'plot_lon_stats_multi',
                  'compute_correlation_pair',
                  'build_selected_correlation_display',
                  'compute_pairwise_correlations',
                  'build_correlation_table'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': 'd24be7c35fa9dc6bd295fccc3b9d2d7aafbf441a453a887a3c0b1e4eb2399b02',
        'late_lon_population': [14, 16],
        'output_types': ['Figure', 'Div', 'DataTable', 'DataTable', 'Figure', 'DataTable', 'DataTable'],
        'outputs': ['d24be7c35fa9dc6bd295fccc3b9d2d7aafbf441a453a887a3c0b1e4eb2399b02',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    'c5ca9e93744f6592413c30b709a0a6b74902f7c66f3297a0417670f6cef4e428',
                    '7448027897cb71a1ae5da99f069f5beaefab65632c7b97316f49082df63ee189',
                    '9050897355e7bfbbb3e0a200e4b21348b4c2443988c310cd84e2c8a5c322980b'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 106,
                      'dims': [2],
                      'items': '6ee87ad0ec5b612129a39d5c4b15c257540281b6827cfa5ef81514c1e584a1b7',
                      'keys': '0b8f5bf19f60a5013cbac6c4d8bec8583b9c708c949c8105de327309c3445c7a'},
        'results': [['add_lon_nodes', 'e53d0cda3a49fb47c4af4e627c76ec30dc11c663bc3d9e3613629c4e935ed0a4'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27'],
                    ['add_lon_nodes', '40b1c20af27a3e8764616c8f86e63ed540e906418816f8da61a1fcc779af8124'],
                    ['compute_node_feasibility_error',
                     'ebf1f54a049df6bf9d043dd7a63f0c00fe904f70d0064948cc1cab355cec3bcd'],
                    ['add_lon_nodes', '98eccbe5a7616217dcdc2edf4d37731bd36a089e0420676d6092906f2f047d1b'],
                    ['compute_node_feasibility_error',
                     '745a1af93c6e4eb03d7fced17ebbdc201514578056530a107801006a9940a433'],
                    ['compute_node_feasibility_error',
                     'a65e6b80659fd91bc200560249e2b02a0118adb97ceed6c3af78478e1bc99513'],
                    ['compute_correlation_pair',
                     '3e4d14258a70ac7056632569e23d4f6236e6f7900c65bf36a666c7411ddd20e1'],
                    ['compute_pairwise_correlations',
                     '98ea4ccf070634dbf04ea36f060c9b3a0cc259929cba71ac947891b91884dd2d']],
        'stdout': '7ed9a97abdfaafe0392922b580283dcc4ad29370a18fc461852fa9914ef04b0d',
        'stdout_lines': 28,
        'steps': [['add_prior_noise_stn_v5',
                   '41bb78ed4e7b96d770eb78c6059e786507f22cc6cc9aae3046be8085eaad51bb',
                   'af6a4598614101f1b2dc0ffccb8e5224b1d842688d5da3e0f67316e68bc55983',
                   40],
                  ['add_prior_noise_stn_v5',
                   'c17c3f14e77c2181ed2961f2f707b38b442885f6afeb348a075575168bdf866d',
                   '63e944ea5be8836301fa96f2e9b878dae379ec69d7fa618e53538782eb908c4d',
                   74],
                  ['debug_mo_counts',
                   'c17c3f14e77c2181ed2961f2f707b38b442885f6afeb348a075575168bdf866d',
                   '63e944ea5be8836301fa96f2e9b878dae379ec69d7fa618e53538782eb908c4d',
                   74],
                  ['add_lon_nodes',
                   '0e65262576c57717b7faeaa9511533c79304ef0c8e5a882ac6fe9e30c5ff854d',
                   '63e944ea5be8836301fa96f2e9b878dae379ec69d7fa618e53538782eb908c4d',
                   106],
                  ['add_lon_edges',
                   '0e65262576c57717b7faeaa9511533c79304ef0c8e5a882ac6fe9e30c5ff854d',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['style_nodes',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['calculate_lon_statistics',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['calculate_positions',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['build_all_traces',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['create_axis_settings',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['add_lon_nodes',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['compute_node_feasibility_error',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['add_lon_nodes',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['compute_node_feasibility_error',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106],
                  ['compute_node_feasibility_error',
                   '5a11a9c38d3663ec83ba142245b442fb9114ed8783e810852b07feb68523543c',
                   '2e0cd08444eb8b175964c143953d2bfef7ce9442bb0826d788e636d8df976950',
                   106]]},
 'M4': {'advanced_traces': ['Advanced misjudgement: increasing noise'],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'series_idx',
                                        'size',
                                        'sol_idx',
                                        'solution',
                                        'start_node',
                                        'type',
                                        'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color',
                                             'edge_type',
                                             'evals',
                                             'is_noisy',
                                             'norm_weight',
                                             'weight'],
                          'edge_count': 145,
                          'edge_types': ['LON', 'Noise_SO', 'NoisyPath_SO', 'STN_SO'],
                          'edges': 'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                          'graph_attrs': {'dict': []},
                          'node_count': 106,
                          'node_ids': '912af41e287d0269e9a4e04326868e9316e1f491274a0a57176751c996634c9a',
                          'nodes': 'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                          'types': ['LON', 'STN_SO', 'STN_SO_Noise']},
        'calls': ['parse_callback_inputs',
                  'add_prior_noise_stn_algo_pov',
                  'generate_run_summary_string',
                  'add_prior_noise_stn_algo_pov',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'plot_lon_stats'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': '253437072a5ae97b76b4577f33ec2c4a97658b6b1fd2a369cec13177f3db96a8',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'DataTable', 'DataTable', 'Figure', 'Div', 'Div'],
        'outputs': ['3c6f5df16a66ebf9f8356f963cc1a446fa31ccc46d2509e722c35abb0d9f9c7a',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 106,
                      'dims': [2],
                      'items': '23d52d41c8b6ccc7dbd9c2000d7d53e0c0b88721f31a8a343facbfff5502237f',
                      'keys': '912af41e287d0269e9a4e04326868e9316e1f491274a0a57176751c996634c9a'},
        'results': [['add_lon_nodes', '5b203623b495413b27691bbccef7a9a339b9c8a7a6e65bec4d34fedc3821afb8'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27']],
        'stdout': 'be97a18f459a42f7b8fa961d21ca58b9356d68a2407093d1428645662e660869',
        'stdout_lines': 27,
        'steps': [['add_prior_noise_stn_algo_pov',
                   '70cccdb304ee883ecdb2cd2c2e479a7fe63aa76a85cccfedf7774aefba2351a8',
                   'c222ce7bc933168e7795059520d8d6656fdfea8bd40b29ebd1977ea9cf3f76c9',
                   40],
                  ['add_prior_noise_stn_algo_pov',
                   '5050826f95485e0bbfc86032ebe6c74577d4fbdd49cae26fe5245b1f3b200890',
                   'd7e3a5fd3d8bf206e6fd12237b5d3c0e818f118bb93fd931e42cf7778876997d',
                   74],
                  ['debug_mo_counts',
                   '5050826f95485e0bbfc86032ebe6c74577d4fbdd49cae26fe5245b1f3b200890',
                   'd7e3a5fd3d8bf206e6fd12237b5d3c0e818f118bb93fd931e42cf7778876997d',
                   74],
                  ['add_lon_nodes',
                   '6ffc6bf1c016756e78bcc940536092aa3a64a84c8babc58235e273f001a343f0',
                   'd7e3a5fd3d8bf206e6fd12237b5d3c0e818f118bb93fd931e42cf7778876997d',
                   106],
                  ['add_lon_edges',
                   '6ffc6bf1c016756e78bcc940536092aa3a64a84c8babc58235e273f001a343f0',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106],
                  ['style_nodes',
                   'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106],
                  ['calculate_lon_statistics',
                   'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106],
                  ['calculate_positions',
                   'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106],
                  ['build_all_traces',
                   'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106],
                  ['create_axis_settings',
                   'd39d80f56b17e42e646f15edf6631c1f709b0f51381b213ac347915c13f06114',
                   'ada69d46cb5619e12edc206207e95752a835debbb603c26d555b5d4b76a991f8',
                   106]]},
 'M5': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'size',
                                        'solution',
                                        'start_node',
                                        'step',
                                        'type'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'evals', 'norm_weight', 'weight'],
                          'edge_count': 142,
                          'edge_types': ['LON', 'Noise', 'NoisyPath_SO', 'STN'],
                          'edges': 'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                          'graph_attrs': {'dict': []},
                          'node_count': 100,
                          'node_ids': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6',
                          'nodes': 'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                          'types': ['LON', 'None', 'STN']},
        'calls': ['parse_callback_inputs',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'compute_node_feasibility_error',
                  'plot_lon_stats',
                  'compute_correlation_pair',
                  'build_selected_correlation_display',
                  'compute_pairwise_correlations',
                  'build_correlation_table'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': '1a43830f75655cc8159cbe6c362928c5a9fc5f7e3d145b96449b9a547a4953b2',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'DataTable', 'DataTable', 'Figure', 'DataTable', 'DataTable'],
        'outputs': ['1a43830f75655cc8159cbe6c362928c5a9fc5f7e3d145b96449b9a547a4953b2',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '25a48d5fc09c72f593b41ed1467aee2f3a584f2acd481645e089f08d8c4403c3',
                    '89a83f81d65cf607116279e2711f63ab05f2a8cb1cf1acb4a1f7cbaf40832ada',
                    '26c421d6c5d00d8304bbfe204e3f4e12ecf48aa0036bca62490612c1d9a913f1',
                    '77ae57c1a842fff62cf619d7221505ce5d7bd8436564e753d8974791fe8342f8'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 100,
                      'dims': [2],
                      'items': 'fc5056aeb7d582ac47cf2071f839bec1ba15aa79045136a17950d8ecae188a79',
                      'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
        'results': [['add_stn_trajectories',
                     'aa02a7a7ff9ccd1933989923b4eb0e52b8179ec92ec48c68fecb571321d64f71'],
                    ['add_stn_trajectories',
                     'e68ae1bd1feb4ff4c288bc5274d4c1f9a04bef8b71c974e8c0860dda121e6a84'],
                    ['add_lon_nodes', '5b203623b495413b27691bbccef7a9a339b9c8a7a6e65bec4d34fedc3821afb8'],
                    ['calculate_lon_statistics',
                     '4dd7c58a247ae227e26c3322e1a93740c40e8b2adf24d6c668f45677e010f0a0'],
                    ['compute_node_feasibility_error',
                     'fd1f6ffd22f3d480ed2cec62a96f7b76e4ba0aedd0cdfd21840e7924b38d2c5f'],
                    ['compute_correlation_pair',
                     '9e63920637c3a8e1134aaa7ad5f514d2c2f0b1ade384804e9f2b7869dd5e2df8'],
                    ['compute_pairwise_correlations',
                     '55e72bd4ba3f881e3786f960baf98b32e69132724ce38c8bffabd4fd972e5125']],
        'stdout': '5c8b9cfb3fabea4cf4b52ea5153b6cfcfb352af275080a8b8912783e33713e71',
        'stdout_lines': 30,
        'steps': [['add_stn_trajectories',
                   '2e375aaf45ddf6f7ae88303374ff7a8cd46081ed84fdac26dbbdf10ba6f27594',
                   'bafbffe9076328098d1f746c7e2611c0bcd22ee39103515d0fbd2806e74d5235',
                   38],
                  ['add_stn_trajectories',
                   '838323a1c16a6062445b298b95b50916b2b7f9262724e7f79d3832de09e7096f',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['debug_mo_counts',
                   '838323a1c16a6062445b298b95b50916b2b7f9262724e7f79d3832de09e7096f',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['add_lon_nodes',
                   'd87764cc3698aa9f384504337f686e0505b56ad375830069aa354f432765ddc2',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   100],
                  ['add_lon_edges',
                   'd87764cc3698aa9f384504337f686e0505b56ad375830069aa354f432765ddc2',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['style_nodes',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['calculate_lon_statistics',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['calculate_positions',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['build_all_traces',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['create_axis_settings',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100],
                  ['compute_node_feasibility_error',
                   'f62069662c39199807494736cd798d1ed67f6a18721bb43ada704edae7aa665d',
                   'dc16359022648b9d2fa74ea536daa4a41cf2ee11b6c5fd7936c76e91a8baf2eb',
                   100]]},
 'M6': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color',
                                        'color_val',
                                        'fitness',
                                        'front_size',
                                        'front_solutions',
                                        'gen_idx',
                                        'guide_series',
                                        'hypervolume',
                                        'is_noisy',
                                        'run_idx',
                                        'size',
                                        'solution',
                                        'type',
                                        'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'is_noisy', 'norm_weight', 'weight'],
                          'edge_count': 94,
                          'edge_types': ['LON', 'Noise_MO', 'STN_MO'],
                          'edges': '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                          'graph_attrs': {'dict': []},
                          'node_count': 107,
                          'node_ids': '890dc1f655465d13c73ea2eb4cb6d7a2bb4362d07d1557932a44a12d473f5b06',
                          'nodes': 'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                          'types': ['LON', 'STN_MO', 'STN_MO_Noise', 'guide']},
        'calls': ['parse_callback_inputs',
                  'add_mo_fronts',
                  'add_mo_fronts',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  '_add_guide_nodes',
                  'calculate_positions',
                  'build_all_traces',
                  'create_guide_traces',
                  'create_axis_settings',
                  'create_figure',
                  'plot_lon_stats'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': 'b942d7609d694793c01de4e495800075f0e7c5d0bb0a64590f9c1892b78c25a7',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'Div', 'DataTable', 'Figure', 'Div', 'Div'],
        'outputs': ['b942d7609d694793c01de4e495800075f0e7c5d0bb0a64590f9c1892b78c25a7',
                    '720a4bc895d650f1f5956a0991d4ab3122f1d190f1f1b3dbf487270c86de98a0',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 54,
                      'dims': [2],
                      'items': '87c81d6f86a5a4c9bb4705c7636282495db70adcb62ec49925b2a4a97bee2c10',
                      'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
        'results': [['add_mo_fronts', '003d38f143c4063309dacc1c00906ef2180075421e9f0e837895c33e98c66109'],
                    ['add_mo_fronts', '842a43775f0ed400f57ce8aa6ebf0d2d4ce8e8cdf70cff963308a0b64307b093'],
                    ['add_lon_nodes', '5b203623b495413b27691bbccef7a9a339b9c8a7a6e65bec4d34fedc3821afb8'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27']],
        'stdout': '7aa9fe4c9d032dc66e4d99c09c8022d548964967bd6ed49f2ea1f57120b569ae',
        'stdout_lines': 74,
        'steps': [['add_mo_fronts',
                   '099a491ad4f0c2edcc9fc228c4889423dfb4c1d2c82c70331efb846a26d0a5bd',
                   '9ef1368bd8ded29092b5165e337f063db00e3884a05d28d5a50c193e3fc627ca',
                   40],
                  ['add_mo_fronts',
                   '6174ad78cb8ea92d778954051e40bfa37d75e1ce6154640da008ffa88ee826bf',
                   '6fd7369cc5af76759525437d89988d60b493bbf037611aba9f473bd491e23fd4',
                   54],
                  ['debug_mo_counts',
                   '6174ad78cb8ea92d778954051e40bfa37d75e1ce6154640da008ffa88ee826bf',
                   '6fd7369cc5af76759525437d89988d60b493bbf037611aba9f473bd491e23fd4',
                   54],
                  ['add_lon_nodes',
                   'cf156e48af8fcb784aaa1baa54dc76ba4f22dc8cdea2c3b2339c6bfcc9b96f06',
                   '6fd7369cc5af76759525437d89988d60b493bbf037611aba9f473bd491e23fd4',
                   86],
                  ['add_lon_edges',
                   'cf156e48af8fcb784aaa1baa54dc76ba4f22dc8cdea2c3b2339c6bfcc9b96f06',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   86],
                  ['style_nodes',
                   'ccbba2d03d5bce81e23316bedb241cfd76d765b8a480a8912f6c1b91393d53a0',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   86],
                  ['calculate_lon_statistics',
                   'ccbba2d03d5bce81e23316bedb241cfd76d765b8a480a8912f6c1b91393d53a0',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   86],
                  ['_add_guide_nodes',
                   'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   107],
                  ['calculate_positions',
                   'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   107],
                  ['build_all_traces',
                   'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   107],
                  ['create_guide_traces',
                   'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   107],
                  ['create_axis_settings',
                   'a228ffb0b7d2f1a426d9579045e12d548636a414048a01c65e460769e653fdbb',
                   '5c87decbfac8ad2f6823892cd07d07410aa1b5b925b39af2fbe2fdd89c973c50',
                   107]]},
 'M7': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color', 'fitness', 'size', 'solution', 'type', 'weight'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'norm_weight', 'weight'],
                          'edge_count': 42,
                          'edge_types': ['LON'],
                          'edges': '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                          'graph_attrs': {'dict': []},
                          'node_count': 32,
                          'node_ids': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9',
                          'nodes': '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                          'types': ['LON']},
        'calls': ['parse_callback_inputs',
                  'debug_mo_counts',
                  'add_lon_nodes',
                  'add_lon_edges',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'compute_node_feasibility_error',
                  'plot_lon_stats',
                  'compute_correlation_pair',
                  'build_selected_correlation_display',
                  'compute_pairwise_correlations',
                  'build_correlation_table'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': '1c007bf4e96cb39fda509fa3d289dc8d9fc244576b78201aa22e6fed88420051',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'Div', 'DataTable', 'Figure', 'DataTable', 'DataTable'],
        'outputs': ['1c007bf4e96cb39fda509fa3d289dc8d9fc244576b78201aa22e6fed88420051',
                    '2a7834318e9f4c782986789775b1d924b695139599506ebdc0e780424194da01',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    '9cfe0785362322778667b75f8ed3aeafdaaa68130a0516e59de6ae1a14f87508',
                    '89a83f81d65cf607116279e2711f63ab05f2a8cb1cf1acb4a1f7cbaf40832ada',
                    '26c421d6c5d00d8304bbfe204e3f4e12ecf48aa0036bca62490612c1d9a913f1',
                    '77ae57c1a842fff62cf619d7221505ce5d7bd8436564e753d8974791fe8342f8'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 32,
                      'dims': [2],
                      'items': '5e2abd58c3b16cc60abf7fa45797fe7bf403b5ca5d8e8a9d1e6aaea659f784aa',
                      'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
        'results': [['add_lon_nodes', '5b203623b495413b27691bbccef7a9a339b9c8a7a6e65bec4d34fedc3821afb8'],
                    ['calculate_lon_statistics',
                     '709254c29437d7904b0533e25e282d787444d6d3e1f5dd6384f12079ccb7ee27'],
                    ['compute_node_feasibility_error',
                     'fd1f6ffd22f3d480ed2cec62a96f7b76e4ba0aedd0cdfd21840e7924b38d2c5f'],
                    ['compute_correlation_pair',
                     '9e63920637c3a8e1134aaa7ad5f514d2c2f0b1ade384804e9f2b7869dd5e2df8'],
                    ['compute_pairwise_correlations',
                     '55e72bd4ba3f881e3786f960baf98b32e69132724ce38c8bffabd4fd972e5125']],
        'stdout': '5329b2c0f35152021de84093fdc9a015cb3f04eab3eb62cd535ff41536d878bc',
        'stdout_lines': 27,
        'steps': [['debug_mo_counts',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['add_lon_nodes',
                   '18db71d37c179f0a37450475e49f5e0e505059aa1c0f70d4b03ac86bb353b50c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   32],
                  ['add_lon_edges',
                   '18db71d37c179f0a37450475e49f5e0e505059aa1c0f70d4b03ac86bb353b50c',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['style_nodes',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['calculate_lon_statistics',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['calculate_positions',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['build_all_traces',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['create_axis_settings',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32],
                  ['compute_node_feasibility_error',
                   '36b2afcb82bef06929f0e633571811db753765f8948c4df85772d1cde585881b',
                   '5ac985f56da1810c57411a0f1f459b09242cc58aeefc693a842248eda1d4b002',
                   32]]},
 'M8': {'advanced_traces': [],
        'before_layout': {'attr_keys': ['color',
                                        'count_estimated_adopted',
                                        'count_estimated_discarded',
                                        'end_node',
                                        'estimated_fitness_adopted',
                                        'estimated_fitness_discarded',
                                        'evals',
                                        'fitness',
                                        'fitness_boxplot_stats',
                                        'is_noisy',
                                        'iterations',
                                        'run_idx',
                                        'size',
                                        'solution',
                                        'start_node',
                                        'step',
                                        'type'],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': ['color', 'edge_type', 'evals', 'weight'],
                          'edge_count': 100,
                          'edge_types': ['Noise', 'NoisyPath_SO', 'STN'],
                          'edges': 'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                          'graph_attrs': {'dict': []},
                          'node_count': 68,
                          'node_ids': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e',
                          'nodes': '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                          'types': ['None', 'STN']},
        'calls': ['parse_callback_inputs',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'add_stn_trajectories',
                  'generate_run_summary_string',
                  'debug_mo_counts',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'plot_lon_stats'],
        'diagnostic': False,
        'error': None,
        'figure_semantic': 'f254ec482b446b675fdfd3c82f91cde0b6a0ad7173971c8600f8652c9bc9cedd',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'DataTable', 'Div', 'Figure', 'Div', 'Div'],
        'outputs': ['f254ec482b446b675fdfd3c82f91cde0b6a0ad7173971c8600f8652c9bc9cedd',
                    '3d0ec7f26009b857277071452369a4d1ba6a22b3a98b6534c19edc02d951fe08',
                    'd29f26ea713e4ad2b5079a73081c9fddb3084f5b0ba8f0f5fadb86c9af566140',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 68,
                      'dims': [2],
                      'items': '7e6bb6d0b49fa6ecae9737f16e8fcdd1fa25bba7d225c21215eb3b00093f3d10',
                      'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
        'results': [['add_stn_trajectories',
                     'aa02a7a7ff9ccd1933989923b4eb0e52b8179ec92ec48c68fecb571321d64f71'],
                    ['add_stn_trajectories',
                     'e68ae1bd1feb4ff4c288bc5274d4c1f9a04bef8b71c974e8c0860dda121e6a84'],
                    ['calculate_lon_statistics',
                     '7acc563833b65e51134d754997a7c9abc54882806b587194f0559a102e4ecf6f']],
        'stdout': '47529a1381f7f122d2ed2b1ccf0470588db48af01588e63e1192994104f4d388',
        'stdout_lines': 18,
        'steps': [['add_stn_trajectories',
                   '4f51b2ede901521599fdf7da46f09fdeb6fbcd726c37ad81997361f72ccf1fbc',
                   'bafbffe9076328098d1f746c7e2611c0bcd22ee39103515d0fbd2806e74d5235',
                   38],
                  ['add_stn_trajectories',
                   '442edc0f1912849d7aa087bb04eb845a933127d050d071832fd8ac59e93c7dcf',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['debug_mo_counts',
                   '442edc0f1912849d7aa087bb04eb845a933127d050d071832fd8ac59e93c7dcf',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['style_nodes',
                   '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['calculate_lon_statistics',
                   '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['calculate_positions',
                   '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['build_all_traces',
                   '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68],
                  ['create_axis_settings',
                   '804bd115a9261729d1b3dc33d1318086b53614ba7ca207ad7f164130dcd20be0',
                   'd0670f058f65a56d81eb163e52c596aed04237cf82f72cb598ec2f864dcf47a5',
                   68]]},
 'M9': {'advanced_traces': [],
        'before_layout': {'attr_keys': [],
                          'class': 'MultiDiGraph',
                          'edge_attr_keys': [],
                          'edge_count': 0,
                          'edge_types': [],
                          'edges': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                          'graph_attrs': {'dict': []},
                          'node_count': 0,
                          'node_ids': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                          'nodes': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                          'types': []},
        'calls': ['parse_callback_inputs',
                  'debug_mo_counts',
                  'style_nodes',
                  'calculate_lon_statistics',
                  'calculate_positions',
                  'build_all_traces',
                  'create_axis_settings',
                  'create_figure',
                  'plot_lon_stats'],
        'diagnostic': True,
        'error': None,
        'figure_semantic': '28dfc2ac5fdd9c02bc43c60648369541002c7a323d932d790afb60ca9de8fce4',
        'late_lon_population': [],
        'output_types': ['Figure', 'Div', 'Div', 'Div', 'Figure', 'Div', 'Div'],
        'outputs': ['28dfc2ac5fdd9c02bc43c60648369541002c7a323d932d790afb60ca9de8fce4',
                    '2a7834318e9f4c782986789775b1d924b695139599506ebdc0e780424194da01',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'fd6ff3073259b4da9345d55a36fc9cae9f86da8a0deeb6b31334f6a43ad0f5f2',
                    '3d13a36d5ec57d2c71140c671670051fa75232719db6e8a0930551848eecc95f',
                    'f7745dacf06f9ef6406e834c1a24ba2f936c13e195f7ece8f55c70f6e3ac8712'],
        'positions': {'all_finite': True,
                      'class': 'dict',
                      'count': 0,
                      'dims': [],
                      'items': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                      'keys': '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c'},
        'results': [['calculate_lon_statistics',
                     '7acc563833b65e51134d754997a7c9abc54882806b587194f0559a102e4ecf6f']],
        'stdout': '62bc76f31b7ce14e52b21579babbefc9747c991c0ba796a71a2740d4a687e19c',
        'stdout_lines': 15,
        'steps': [['debug_mo_counts',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['style_nodes',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['calculate_lon_statistics',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['calculate_positions',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['build_all_traces',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0],
                  ['create_axis_settings',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   '6d355a642ffed28c2afbda550638c33deb10c7f463ea5f2ce79a5ee5a8e15a4c',
                   0]]}}

LAYOUTS = {'fps_lmds|lon_only': {'error': None,
                       'smoke_only': True,
                       'structural': [{'all_finite': True,
                                       'class': 'dict',
                                       'count': 32,
                                       'dims': [3],
                                       'error': None,
                                       'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                                      {'all_finite': True,
                                       'class': 'dict',
                                       'count': 32,
                                       'dims': [3],
                                       'error': None,
                                       'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'}]},
 'fps_lmds|mixed': {'error': None,
                    'smoke_only': True,
                    'structural': [{'all_finite': True,
                                    'class': 'dict',
                                    'count': 100,
                                    'dims': [3],
                                    'error': None,
                                    'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                                   {'all_finite': True,
                                    'class': 'dict',
                                    'count': 100,
                                    'dims': [3],
                                    'error': None,
                                    'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'}]},
 'fps_lmds|mo': {'error': None,
                 'exact': {'all_finite': True,
                           'class': 'dict',
                           'count': 54,
                           'dims': [2],
                           'error': None,
                           'items': '5b6abe33dce10050b93298cdc3d1893e488c0a2fd66d409c2c3c35413e353a1d',
                           'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
                 'smoke_only': False},
 'fps_lmds|stn_only': {'error': None,
                       'smoke_only': True,
                       'structural': [{'all_finite': True,
                                       'class': 'dict',
                                       'count': 68,
                                       'dims': [3],
                                       'error': None,
                                       'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                                      {'all_finite': True,
                                       'class': 'dict',
                                       'count': 68,
                                       'dims': [3],
                                       'error': None,
                                       'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'}]},
 'hamming_delta_ref|lon_only': {'error': None,
                                'exact': {'all_finite': True,
                                          'class': 'dict',
                                          'count': 32,
                                          'dims': [2],
                                          'error': None,
                                          'items': '3a8b86be2a96c35d140c79907bcd28adcaeefe1c82819a1fceaf6e782f101049',
                                          'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                                'smoke_only': False},
 'hamming_delta_ref|mixed': {'error': None,
                             'exact': {'all_finite': True,
                                       'class': 'dict',
                                       'count': 100,
                                       'dims': [2],
                                       'error': None,
                                       'items': 'aec9f5a2fa2d33d6fd8541348c4cbe6484e1eb30c51d478107512d53ddb3507e',
                                       'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                             'smoke_only': False},
 'hamming_delta_ref|mo': {'error': None,
                          'exact': {'all_finite': True,
                                    'class': 'dict',
                                    'count': 54,
                                    'dims': [2],
                                    'error': None,
                                    'items': '87c81d6f86a5a4c9bb4705c7636282495db70adcb62ec49925b2a4a97bee2c10',
                                    'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
                          'smoke_only': False},
 'hamming_delta_ref|stn_only': {'error': None,
                                'exact': {'all_finite': True,
                                          'class': 'dict',
                                          'count': 68,
                                          'dims': [2],
                                          'error': None,
                                          'items': 'a4ec381203e348d391e22e3524fdc7d5e42498fe3ae2152bf50cf87b16a85198',
                                          'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                                'smoke_only': False},
 'kamada_kawai|lon_only': {'error': None,
                           'exact': {'all_finite': True,
                                     'class': 'dict',
                                     'count': 32,
                                     'dims': [2],
                                     'error': None,
                                     'items': '4cd7a5906b3aa27707b126a30ce010b62e2b338b47f84cae7290f3c08003306e',
                                     'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                           'smoke_only': False},
 'kamada_kawai|mixed': {'error': None,
                        'exact': {'all_finite': True,
                                  'class': 'dict',
                                  'count': 100,
                                  'dims': [2],
                                  'error': None,
                                  'items': '095749ad61253ce7e800402c830caffe46fb74e2d4e380f7bd4fe961893ce906',
                                  'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                        'smoke_only': False},
 'kamada_kawai|mo': {'error': None,
                     'exact': {'all_finite': True,
                               'class': 'dict',
                               'count': 54,
                               'dims': [2],
                               'error': None,
                               'items': '3c6081ff275cb7070d704868872d5280865344d35c58d91bbc513c02dde0c60f',
                               'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
                     'smoke_only': False},
 'kamada_kawai|stn_only': {'error': None,
                           'exact': {'all_finite': True,
                                     'class': 'dict',
                                     'count': 68,
                                     'dims': [2],
                                     'error': None,
                                     'items': '1d5590c11a93ebe1f89178210ca423e7f51ea746a5681c5ffb219ad7251d5cb9',
                                     'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                           'smoke_only': False},
 'lon_lmds|lon_only': {'error': None,
                       'exact': {'all_finite': True,
                                 'class': 'dict',
                                 'count': 32,
                                 'dims': [2],
                                 'error': None,
                                 'items': 'fc9a144ee8d731e781f073c2ce9523cb4eedb04365b1fb1e6238bb0a3828210b',
                                 'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                       'smoke_only': False},
 'lon_lmds|mixed': {'error': None,
                    'exact': {'all_finite': True,
                              'class': 'dict',
                              'count': 100,
                              'dims': [2],
                              'error': None,
                              'items': '3fa4a6e291140ab98a9f54c7d3b412e7fdfe8473fb18017257e4694b7f9c8210',
                              'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                    'smoke_only': False},
 'lon_lmds|mo': {'error': None,
                 'exact': {'all_finite': True,
                           'class': 'dict',
                           'count': 54,
                           'dims': [2],
                           'error': None,
                           'items': '87c81d6f86a5a4c9bb4705c7636282495db70adcb62ec49925b2a4a97bee2c10',
                           'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
                 'smoke_only': False},
 'lon_lmds|stn_only': {'error': 'IndexError',
                       'exact': {'error': 'IndexError',
                                 'message': 'arrays used as indices must be of integer (or boolean) type'},
                       'smoke_only': False},
 'mds|lon_only': {'error': None,
                  'exact': {'all_finite': True,
                            'class': 'dict',
                            'count': 32,
                            'dims': [2],
                            'error': None,
                            'items': '5e2abd58c3b16cc60abf7fa45797fe7bf403b5ca5d8e8a9d1e6aaea659f784aa',
                            'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                  'smoke_only': False},
 'mds|mixed': {'error': None,
               'exact': {'all_finite': True,
                         'class': 'dict',
                         'count': 100,
                         'dims': [2],
                         'error': None,
                         'items': '14dbd24d5ab9684bb6aaff2af16aebaf8dce15a1bdf414b3009a047216b77fb2',
                         'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
               'smoke_only': False},
 'mds|mo': {'error': None,
            'exact': {'all_finite': True,
                      'class': 'dict',
                      'count': 54,
                      'dims': [2],
                      'error': None,
                      'items': '87c81d6f86a5a4c9bb4705c7636282495db70adcb62ec49925b2a4a97bee2c10',
                      'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
            'smoke_only': False},
 'mds|stn_only': {'error': None,
                  'exact': {'all_finite': True,
                            'class': 'dict',
                            'count': 68,
                            'dims': [2],
                            'error': None,
                            'items': '7e6bb6d0b49fa6ecae9737f16e8fcdd1fa25bba7d225c21215eb3b00093f3d10',
                            'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                  'smoke_only': False},
 'r_lmds|lon_only': {'error': None,
                     'exact': {'all_finite': True,
                               'class': 'dict',
                               'count': 32,
                               'dims': [2],
                               'error': None,
                               'items': 'ce703a9b2a952d8133a1f6724dc9f4ac6c5c02c973eb743e3247f8c9ade46deb',
                               'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                     'smoke_only': False},
 'r_lmds|mixed': {'error': None,
                  'exact': {'all_finite': True,
                            'class': 'dict',
                            'count': 100,
                            'dims': [2],
                            'error': None,
                            'items': '6d50943b82bb8f8e9c92275be572aa787c537ff00e4a345f2f23e0979cf2a040',
                            'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                  'smoke_only': False},
 'r_lmds|mo': {'error': None,
               'exact': {'all_finite': True,
                         'class': 'dict',
                         'count': 54,
                         'dims': [2],
                         'error': None,
                         'items': '556e929d5da0323fd109af3f5a41a16400cec0c2163ab1d693c985fdeb254bb5',
                         'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
               'smoke_only': False},
 'r_lmds|stn_only': {'error': None,
                     'exact': {'all_finite': True,
                               'class': 'dict',
                               'count': 68,
                               'dims': [2],
                               'error': None,
                               'items': '42e792fe992e63c425ac87d4f0705f7b538422b0e5472736c119107b616e1fa4',
                               'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                     'smoke_only': False},
 'raw|lon_only': {'error': None,
                  'exact': {'all_finite': True,
                            'class': 'dict',
                            'count': 32,
                            'dims': [2],
                            'error': None,
                            'items': '306f1c558e5e6d7ad8da1be7b0f93d3937a11bf6372ec8aae8e72bbdc58a229f',
                            'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                  'smoke_only': False},
 'raw|mixed': {'error': None,
               'exact': {'all_finite': True,
                         'class': 'dict',
                         'count': 100,
                         'dims': [2],
                         'error': None,
                         'items': 'df4b6362367787f023c61f8240fff9c49ca73a81ab1596f7c39c1f7293b69777',
                         'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
               'smoke_only': False},
 'raw|mo': {'error': None,
            'exact': {'all_finite': True,
                      'class': 'dict',
                      'count': 54,
                      'dims': [2],
                      'error': None,
                      'items': '87c81d6f86a5a4c9bb4705c7636282495db70adcb62ec49925b2a4a97bee2c10',
                      'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
            'smoke_only': False},
 'raw|stn_only': {'error': None,
                  'exact': {'all_finite': True,
                            'class': 'dict',
                            'count': 68,
                            'dims': [2],
                            'error': None,
                            'items': '853a7810d7b310d3bb7b00878e78a220fc0a75db2463b5c268a1a100e724b7e3',
                            'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                  'smoke_only': False},
 'spring|lon_only': {'error': None,
                     'smoke_only': True,
                     'structural': [{'all_finite': True,
                                     'class': 'dict',
                                     'count': 32,
                                     'dims': [3],
                                     'error': None,
                                     'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                                    {'all_finite': True,
                                     'class': 'dict',
                                     'count': 32,
                                     'dims': [3],
                                     'error': None,
                                     'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'}]},
 'spring|mixed': {'error': None,
                  'smoke_only': True,
                  'structural': [{'all_finite': True,
                                  'class': 'dict',
                                  'count': 100,
                                  'dims': [3],
                                  'error': None,
                                  'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                                 {'all_finite': True,
                                  'class': 'dict',
                                  'count': 100,
                                  'dims': [3],
                                  'error': None,
                                  'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'}]},
 'spring|mo': {'error': None,
               'exact': {'all_finite': True,
                         'class': 'dict',
                         'count': 54,
                         'dims': [2],
                         'error': None,
                         'items': '3c6081ff275cb7070d704868872d5280865344d35c58d91bbc513c02dde0c60f',
                         'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
               'smoke_only': False},
 'spring|stn_only': {'error': None,
                     'smoke_only': True,
                     'structural': [{'all_finite': True,
                                     'class': 'dict',
                                     'count': 68,
                                     'dims': [3],
                                     'error': None,
                                     'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                                    {'all_finite': True,
                                     'class': 'dict',
                                     'count': 68,
                                     'dims': [3],
                                     'error': None,
                                     'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'}]},
 'tsne|lon_only': {'error': None,
                   'exact': {'all_finite': True,
                             'class': 'dict',
                             'count': 32,
                             'dims': [2],
                             'error': None,
                             'items': 'c5a03f92b89771c68039d5538376103a36f492ef471984ee3bd93972901de0a2',
                             'keys': 'c95aaf905627b3c0e1fe9f433cda1b62136f098a37b884324072556e89bab4b9'},
                   'smoke_only': False},
 'tsne|mixed': {'error': None,
                'exact': {'all_finite': True,
                          'class': 'dict',
                          'count': 100,
                          'dims': [2],
                          'error': None,
                          'items': 'fc5056aeb7d582ac47cf2071f839bec1ba15aa79045136a17950d8ecae188a79',
                          'keys': '9e9f4a365c9bc9b725a83e4b7a014cbed568df2b46b62afb0b56ba6186ac0ba6'},
                'smoke_only': False},
 'tsne|mo': {'error': None,
             'exact': {'all_finite': True,
                       'class': 'dict',
                       'count': 54,
                       'dims': [2],
                       'error': None,
                       'items': '10624a46006eb986e3352b3d85e36795efdac5299beac5593e3978b20f8c07a4',
                       'keys': 'd2bd85fca97a336174a42a6c4d6904c48f534988707c7af3c112baf876e44b94'},
             'smoke_only': False},
 'tsne|stn_only': {'error': None,
                   'exact': {'all_finite': True,
                             'class': 'dict',
                             'count': 68,
                             'dims': [2],
                             'error': None,
                             'items': '224ddc67456771cbada5daff89571642a31a52f9d5d3079f03dd9e7097a69dc6',
                             'keys': '63382f8308838b51302fa7f6d89b2ed18cd94c0049a8e45ea4906664400d724e'},
                   'smoke_only': False}}

# ------------------------------------------------------- pre-move module -> intended post-move module

MODULE_MOVES = {
    "visualization/config.py": "viz/config.py",
    "visualization/node_positioning.py": "viz/layout.py",
    "visualization/node_styling.py": "viz/styling.py",
    "visualization/trace_builder.py": "viz/traces.py",
    "visualization/statistics.py": "analysis/graph_stats.py",
    "visualization/graph_builder.py": "viz/graph/stn.py",
    "visualization/lon_stats_plots.py": "viz/plots/lon_stats.py",
}

# Definitions whose post-move module differs from their module's default destination: the LON half of
# the graph-population split, and the two Dash table builders.
SPLIT_OVERRIDES = {
    "add_lon_nodes": "viz/graph/lon.py",
    "add_lon_edges": "viz/graph/lon.py",
    "build_correlation_table": "dashboard/components.py",
    "build_selected_correlation_display": "dashboard/components.py",
}


def _post_module(name: str, pre_module: str) -> str:
    if name in SPLIT_OVERRIDES:
        return SPLIT_OVERRIDES[name]
    if pre_module in MODULE_MOVES:
        return MODULE_MOVES[pre_module]
    if pre_module.startswith("plotting/"):
        return "viz/plots/" + pre_module[len("plotting/"):]
    raise AssertionError(f"no post-Stage-10 location defined for {pre_module}")


def allowed_modules(name: str) -> tuple:
    pre = DEFINITION_MODULES[name]
    post = _post_module(name, pre)
    return (pre, post) if ALLOW_PRE_LOCATIONS else (post,)


# The graph-population functions of graph_builder.py, by destination (plan Stage 10 §6).
STN_POPULATION = (
    "generate_run_summary_string", "print_hamming_transitions", "add_stn_trajectories",
    "add_mo_fronts", "add_prior_noise_stn_v4", "add_prior_noise_stn_v5",
    "add_prior_noise_stn_algo_pov", "debug_mo_counts",
)
LON_POPULATION = ("add_lon_nodes", "add_lon_edges")

_PROBE = r'''
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import os
import random
import sys
import textwrap
from pathlib import Path

SOURCE_ROOT = Path(__SOURCE_ROOT__)
WORKSPACE = Path(__WORKSPACE__)
KP_PID = "f1_l-d_kp_10_269"

# The source tree under inspection comes first; for the test itself this is the workspace.
sys.path.insert(0, str(SOURCE_ROOT / "src"))

spec = importlib.util.spec_from_file_location("_harness_fence", __FENCE__)
fence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fence)
fence.install(str(WORKSPACE))

# Stage 11 redistributes Dashboard.py and DashboardHelpers.py, so their statements are found through
# the shared inventory rather than by path (plan §15.1). BODY_KEYS is their frozen PRE_STAGE_11
# statement order, which `reassemble` walks to rebuild each file as it read before the split.
spec = importlib.util.spec_from_file_location("_dashboard_inventory", __INVENTORY__)
INV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INV)
BODY_KEYS = __BODY_KEYS__

import numpy as np
import networkx as nx
import pandas as pd

# create_figure writes its default output under PLOTS_DIR, which follows NOISYVIS_ROOT.
(Path(os.environ["NOISYVIS_ROOT"]) / "plots").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ canonical forms

def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


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
    if isinstance(obj, (set, frozenset)):
        return {"set": sorted(repr(x) for x in obj)}
    if isinstance(obj, dict):
        return {"dict": [[canon(k), canon(v)] for k, v in obj.items()]}
    return {"other": type(obj).__qualname__, "repr": repr(obj)}


def digest(obj):
    return sha256(json.dumps(canon(obj), separators=(",", ":")))


def fig_digest(fig):
    """Digest of a Plotly figure or Dash component through its own JSON form."""
    return digest(fig.to_plotly_json())


ADVANCED_TRACE_PREFIX = "Advanced misjudgement:"


def advanced_sorted_digest(fig):
    """Digest with ONLY the advanced-misjudgement traces' point tuples sorted.

    Those traces are built by iterating a Python set of node labels in Dashboard.update_plot, so
    their point order follows string hash randomisation. Every other trace, and the graph, edge,
    position and output ordering, is left exactly as produced (I-10b).
    """
    payload = fig.to_plotly_json()
    traces = []
    for trace in payload.get("data", []):
        name = trace.get("name") or ""
        if isinstance(name, str) and name.startswith(ADVANCED_TRACE_PREFIX):
            points = sorted(zip(trace.get("x", []), trace.get("y", []), trace.get("z", [])),
                            key=lambda point: tuple(float(value) for value in point))
            trace = dict(trace)
            trace["x"] = [p[0] for p in points]
            trace["y"] = [p[1] for p in points]
            trace["z"] = [p[2] for p in points]
        traces.append(trace)
    payload = dict(payload)
    payload["data"] = traces
    return digest(payload)


def safe_fig(fn, *args, **kwargs):
    """Current behaviour, whether that is a figure or an exception (nothing is 'fixed' here)."""
    try:
        return fig_digest(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "message": str(exc)[:200]}


# ------------------------------------------------------------------ AST helpers

def _absolutise(node, package):
    """Rewrite relative ImportFrom nodes to absolute modules, so hashes are location-agnostic."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.level:
            parts = package.split(".")
            base = parts[: len(parts) - (child.level - 1)] if child.level > 1 else parts
            child.module = ".".join([*base, child.module] if child.module else base)
            child.level = 0
    return node


def norm_ast_sha256(node, package):
    return sha256(ast.dump(_absolutise(node, package)))


def literal_digest(node):
    """Multiset of every string constant, including f-string literal parts (I-10)."""
    literals = sorted(
        child.value for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )
    return digest(literals)


def obj_ast_sha256(obj):
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    package = getattr(module, "__package__", "") or ""
    node = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]
    return norm_ast_sha256(node, package)


def package_of(path):
    """noisyvis.viz.graph for src/noisyvis/viz/graph/lon.py."""
    parts = path.relative_to(SOURCE_ROOT / "src").parts[:-1]
    return ".".join(parts)


def module_key(path):
    return str(path.relative_to(SOURCE_ROOT / "src" / "noisyvis"))


def import_first(*names):
    last = None
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:  # noqa: PERF203
            last = exc
    raise last


# ------------------------------------------------------------------ 1. definition inventory

PKG = SOURCE_ROOT / "src" / "noisyvis"

# Checkpoint E: the final Stage-10 locations only. The pre-move packages `visualization` and
# `plotting` no longer exist and are no longer accepted.
SCAN_DIRS = ["viz", "analysis"]
SCAN_FILES = ["dashboard/components.py"]
OBSOLETE_PACKAGES = ("noisyvis.visualization", "noisyvis.plotting")

# The legacy Pareto monolith is excluded from the definition inventory until Checkpoint F deletes it.
# It is identified by basename under the plots tree, and nowhere else: a file with any other name, or
# this name outside that tree, is scanned like every other module.
MONOLITH_BASENAME = "plotParetoFrontMain.py"
MONOLITH_DIRS = ("viz/plots/",)
MONOLITH_MODULES = ("viz/plots/" + MONOLITH_BASENAME,)


def is_monolith(key, path):
    return path.name == MONOLITH_BASENAME and key.startswith(MONOLITH_DIRS)


def scan_definitions():
    found, duplicates, monoliths = {}, {}, []
    files = []
    for name in SCAN_DIRS:
        directory = PKG / name
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    for name in SCAN_FILES:
        path = PKG / name
        if path.is_file():
            files.append(path)

    for path in files:
        key = module_key(path)
        if is_monolith(key, path):
            monoliths.append(key)
            continue
        if path.name == "__init__.py":
            continue
        package = package_of(path)
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                names = [t.id for t in node.targets]
                kind = "const"
            else:
                continue
            for name in names:
                record = {
                    "kind": kind,
                    "module": key,
                    "ast": norm_ast_sha256(node, package),
                    "literals": literal_digest(node),
                }
                if kind == "def":
                    args = node.args
                    record["args"] = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
                if name in found:
                    duplicates.setdefault(name, [found[name]["module"]]).append(key)
                found[name] = record
    return found, {name: sorted(modules) for name, modules in duplicates.items()}, sorted(monoliths)


def obsolete_import_references():
    """Executable imports of the pre-move packages anywhere under src/noisyvis (never prose)."""
    found = []
    for path in sorted(PKG.rglob("*.py")):
        package = package_of(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                    module = ".".join([*base, node.module] if node.module else base)
                else:
                    module = node.module or ""
                modules = [module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if any(module == obsolete or module.startswith(obsolete + ".")
                       for obsolete in OBSOLETE_PACKAGES):
                    found.append(f"{module_key(path)}:{node.lineno}: {module}")
    return found


# ------------------------------------------------------------------ 2. registry and aliases

VIZ = import_first("noisyvis.viz", "noisyvis.visualization")
PLOTS = import_first("noisyvis.viz.plots", "noisyvis.plotting")
GRAPH_STATS = import_first("noisyvis.analysis.graph_stats", "noisyvis.visualization.statistics")
LON_STATS = import_first("noisyvis.viz.plots.lon_stats", "noisyvis.visualization.lon_stats_plots")
TABLES = import_first("noisyvis.dashboard.components", "noisyvis.visualization.lon_stats_plots")
PERF = import_first("noisyvis.viz.plots.performance", "noisyvis.plotting.performance")

PARETO_ALIASES = {
    "plotParetoFront": "plot_basic",
    "plotParetoFrontSubplots": "plot_subplots",
    "plotParetoFrontSubplotsMulti": "plot_subplots_multi",
    "PlotparetoFrontSubplotsHighlighted": "plot_subplots_highlighted",
    "plotParetoFrontAnimation": "plot_animation",
    "plotParetoFrontNoisy": "plot_noisy",
    "plotParetoFrontIndVsDist": "plot_ind_vs_dist",
    "plotParetoFrontIGDVsDist": "plot_igd_vs_dist",
    "plotProgressPerMovementRatio": "plot_progress_per_movement",
    "plotMovementCorrelation": "plot_movement_correlation",
    "plotMoveDeltaHistograms": "plot_move_delta_histograms",
    "plotObjectiveVsDecisionScatter": "plot_objective_vs_decision",
}
PERF_ALIASES = {
    "plot2d_line": "plot_line", "plot2d_box": "plot_box", "plot2d_line_mo": "plot_line_mo",
    "plot2d_box_mo": "plot_box_mo", "plot2d_line_evals": "plot_line_evals",
    "plot2d_box_evals": "plot_box_evals", "plot2d_box_penalty": "plot_box_penalty",
    "plot2d_box_misjudgements_so": "plot_box_misjudgements_so",
    "plot2d_box_advanced_misjudgements_so": "plot_box_advanced_misjudgements_so",
}


def registry_report():
    report = {"pareto_keys": list(PLOTS.PARETO_PLOTS), "performance_keys": list(PLOTS.PERFORMANCE_PLOTS)}
    report["pareto"] = {
        key: {"name": fn.__name__, "ast": obj_ast_sha256(fn),
              "lookup_is_same": PLOTS.get_pareto_plot(key) is fn}
        for key, fn in PLOTS.PARETO_PLOTS.items()
    }
    report["performance"] = {
        key: {"name": fn.__name__, "ast": obj_ast_sha256(fn),
              "lookup_is_same": PLOTS.get_performance_plot(key) is fn}
        for key, fn in PLOTS.PERFORMANCE_PLOTS.items()
    }
    report["unknown_key_is_none"] = PLOTS.get_pareto_plot("__no_such_plot__") is None
    # Checkpoint F removed these. Report where each one still appears, rather than comparing
    # getattr(..., None) with getattr(..., None), which would be vacuously True once both are gone.
    pareto_pkg = import_first("noisyvis.viz.plots.pareto", "noisyvis.plotting.pareto")
    report["removed_aliases_present"] = {
        alias: sorted(
            where for where, present in (
                ("plots.__dict__", alias in vars(PLOTS)),
                ("pareto.__dict__", alias in vars(pareto_pkg)),
                ("plots.__all__", alias in getattr(PLOTS, "__all__", ())),
                ("pareto.__all__", alias in getattr(pareto_pkg, "__all__", ())),
            ) if present
        )
        for alias in PARETO_ALIASES
    }
    report["removed_aliases_present"] = {a: w for a, w in report["removed_aliases_present"].items() if w}
    report["alias_names_present"] = sorted(a for a in PARETO_ALIASES if hasattr(PLOTS, a))
    report["canonical_targets_present"] = sorted(
        t for t in set(PARETO_ALIASES.values()) if hasattr(PLOTS, t) and hasattr(pareto_pkg, t))
    report["performance_aliases"] = {
        alias: (getattr(PERF, alias) is getattr(PERF, target))
        for alias, target in PERF_ALIASES.items()
    }
    # The Dashboard dropdown values that reach the registry, read statically.
    components = PKG / "dashboard" / "layout" / "components.py"
    tree = ast.parse(components.read_text())
    dropdown = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {k.arg: k.value for k in node.keywords if k.arg}
        target = keywords.get("id")
        if isinstance(target, ast.Constant) and target.value == "paretoFrontPlotType":
            for option in ast.literal_eval(keywords["options"]):
                dropdown.append(option["value"])
    report["dropdown_values"] = dropdown
    report["monolith_present"] = any((PKG / module).is_file() for module in MONOLITH_MODULES)
    return report


# ------------------------------------------------------------------ 3. 2D figure fixtures

def _front(gen, n_pts, shift):
    solutions = [[(i + j + shift) % 2 for j in range(10)] for i in range(n_pts)]
    noisy = [[float(1.0 + i + gen * 0.5 + shift), float(9.0 - i - gen * 0.25)] for i in range(n_pts)]
    clean = [[float(0.9 + i + gen * 0.5 + shift), float(9.2 - i - gen * 0.25)] for i in range(n_pts)]
    true_solutions = [[(i + j + shift + 1) % 2 for j in range(10)] for i in range(n_pts)]
    return {
        "algo_front_solutions": solutions,
        "algo_front_noisy_fitnesses": noisy,
        "algo_front_clean_fitnesses": clean,
        "algo_front_noisy_hypervolume": float(10.0 + gen * 1.5 + shift),
        "algo_front_clean_hypervolume": float(10.5 + gen * 1.4 + shift),
        "clean_front_solutions": true_solutions,
        "clean_front_fitnesses": clean,
        "clean_front_hypervolume": float(11.0 + gen * 1.3 + shift),
        "gen_idx": gen,
    }


FRONTDATA = [
    [[_front(g, 3 + (g % 2), 0) for g in range(6)], [_front(g, 3, 1) for g in range(6)]],
    [[_front(g, 4, 2) for g in range(5)]],
]
SERIES_LABELS = [["MuPlusLambda", 1.0], ["OnePlusOne", 2.0]]

PARETO_CASES = [
    ("Basic", {}),
    ("Subplots", {}),
    ("SubplotsMulti", {"nruns": 2}),
    ("SubplotsHighlight", {}),
    ("paretoAnimation", {}),
    ("Noisy", {}),
    ("PPM", {}),
]
for _method in ("cumulative", "raw", "mds", "tsne", "isomap"):
    PARETO_CASES.append(("IndVsDist", {"distance_method": _method, "nruns": 1}))
    PARETO_CASES.append(("IGDVsDist", {"distance_method": _method, "nruns": 1}))
for _indicator in ("NoisyHV", "CleanHV", "IGD"):
    PARETO_CASES.append(("MoveCorr", {"IndVsDist_IndType": _indicator, "window": 5}))
    PARETO_CASES.append(("Hist", {"IndVsDist_IndType": _indicator}))
    PARETO_CASES.append(("Scatter", {"IndVsDist_IndType": _indicator}))


def pareto_report():
    report = {}
    for key, kwargs in PARETO_CASES:
        label = key + "|" + json.dumps(kwargs, sort_keys=True)
        fn = PLOTS.get_pareto_plot(key)
        report[label] = safe_fig(fn, FRONTDATA, SERIES_LABELS, **kwargs)
    # empty-input fallbacks
    for key, _ in PARETO_CASES:
        fn = PLOTS.get_pareto_plot(key)
        report["empty|" + key] = safe_fig(fn, [], [])
    report["create_pareto_plot|unknown"] = safe_fig(PLOTS.create_pareto_plot, "__nope__", FRONTDATA, SERIES_LABELS)
    return report


PERF_ROWS = []
for _algo_idx, _algo in enumerate(["MuPlusLambda", "OnePlusOne", "SEMO"]):
    for _noise in (0.0, 1.0, 2.0):
        for _seed in range(5):
            base = 100.0 + 10 * _algo_idx - 5 * _noise + _seed
            PERF_ROWS.append({
                "algo_name": _algo, "noise": _noise, "seed": _seed, "penalty": 10.0 * (1 + _algo_idx % 2),
                "max_fit": base, "min_fit": base - 20.0, "final_fit": base - 2.0,
                "max_fit_noisy": base + 1.5, "min_fit_noisy": base - 21.0, "final_fit_noisy": base - 3.5,
                "evals_to_best": 500 + 10 * _seed, "evals_to_final": 800 + 12 * _seed,
                "evals_to_best_noisy": 520 + 10 * _seed, "evals_to_final_noisy": 830 + 12 * _seed,
                "n_misjudgements": 3 + _seed % 4, "n_increasing_noise": 1 + _seed % 3,
                "n_comparison_misjudgements": 2 + _seed % 2, "n_constraint_misjudgements": _seed % 3,
                "final_true_hv": 50.0 + 3 * _algo_idx - _noise + 0.5 * _seed,
            })
PERF_DF = pd.DataFrame(PERF_ROWS)

PERF_CASES = [
    ("plot_line", {"fitness_mode": "best", "problem_goal": "maximise"}),
    ("plot_line", {"fitness_mode": "final", "problem_goal": "minimise"}),
    ("plot_line", {"fitness_mode": "best_noisy", "problem_goal": "maximise", "xaxis_title": "sigma"}),
    ("plot_line_evals", {"fitness_mode": "final", "show_std": True}),
    ("plot_line_evals", {"fitness_mode": "best", "show_std": False}),
    ("plot_line_mo", {}),
    ("plot_box", {"fitness_mode": "best", "problem_goal": "maximise"}),
    ("plot_box", {"fitness_mode": "final_noisy", "problem_goal": "minimise"}),
    ("plot_box_evals", {"fitness_mode": "final"}),
    ("plot_box_penalty", {"fitness_mode": "best", "problem_goal": "maximise"}),
    ("plot_box_misjudgements_so", {}),
    ("plot_box_advanced_misjudgements_so", {"algo_name": "MuPlusLambda"}),
    ("plot_box_mo", {}),
]


def performance_report():
    report = {}
    for name, kwargs in PERF_CASES:
        fn = getattr(PERF, name)
        label = name + "|" + json.dumps(kwargs, sort_keys=True)
        report[label] = safe_fig(fn, PERF_DF.copy(), **kwargs)
    bare = PERF_DF.drop(columns=["n_misjudgements", "final_true_hv", "penalty", "evals_to_final",
                                 "n_increasing_noise", "n_comparison_misjudgements",
                                 "n_constraint_misjudgements"]).copy()
    for name, kwargs in [("plot_box_misjudgements_so", {}), ("plot_box_mo", {}), ("plot_line_mo", {}),
                         ("plot_box_penalty", {"fitness_mode": "best", "problem_goal": "maximise"}),
                         ("plot_box_evals", {"fitness_mode": "final"}),
                         ("plot_box_advanced_misjudgements_so", {"algo_name": "MuPlusLambda"})]:
        report["missing_columns|" + name] = safe_fig(getattr(PERF, name), bare, **kwargs)
    return report


NODE_STATS_INPUT = [
    {"node": "Local Optimum 1", "neigh_feas": 0.2, "fitness": 100.0, "median": 98.5,
     "error": 1.5, "abs_error": 1.5, "iqr": 4.0},
    {"node": "Local Optimum 2", "neigh_feas": 0.4, "fitness": 110.0, "median": 111.25,
     "error": -1.25, "abs_error": 1.25, "iqr": 3.5},
    {"node": "Local Optimum 3", "neigh_feas": 0.4, "fitness": 120.0, "median": 117.0,
     "error": 3.0, "abs_error": 3.0, "iqr": 6.25},
    {"node": "Local Optimum 4", "neigh_feas": 0.8, "fitness": 130.0, "median": 129.5,
     "error": 0.5, "abs_error": 0.5, "iqr": 2.0},
]
NODE_STATS_BY_SERIES = {1: NODE_STATS_INPUT[:3], 2: NODE_STATS_INPUT[1:], 3: []}


def lon_stats_report():
    report = {}
    axes = [("neigh_feas", "error"), ("neigh_feas", "iqr"), ("fitness", "median"), ("error", "abs_error")]
    for x_key, y_key in axes:
        report[f"scatter|{x_key}|{y_key}"] = safe_fig(LON_STATS.plot_lon_scatter, NODE_STATS_INPUT, x_key, y_key)
        report[f"violin|{x_key}|{y_key}"] = safe_fig(LON_STATS.plot_lon_violin, NODE_STATS_INPUT, x_key, y_key)
        for style in ("scatter", "violin"):
            report[f"stats|{style}|{x_key}|{y_key}"] = safe_fig(
                LON_STATS.plot_lon_stats, NODE_STATS_INPUT, x_key, y_key, style)
            report[f"stats_multi|{style}|{x_key}|{y_key}"] = safe_fig(
                LON_STATS.plot_lon_stats_multi, NODE_STATS_BY_SERIES, x_key, y_key, style)
    report["scatter|empty"] = safe_fig(LON_STATS.plot_lon_scatter, [], "neigh_feas", "error")
    report["violin|empty"] = safe_fig(LON_STATS.plot_lon_violin, [], "neigh_feas", "error")
    report["stats|empty"] = safe_fig(LON_STATS.plot_lon_stats, [], "neigh_feas", "error", "scatter")
    report["stats_multi|empty"] = safe_fig(LON_STATS.plot_lon_stats_multi, {}, "neigh_feas", "error", "violin")
    report["stats_multi|all_empty"] = safe_fig(
        LON_STATS.plot_lon_stats_multi, {1: [], 2: []}, "neigh_feas", "error", "scatter")
    report["defaults"] = {
        "AXIS_OPTIONS": canon(LON_STATS.AXIS_OPTIONS), "AXIS_LABELS": canon(LON_STATS.AXIS_LABELS),
        "DEFAULT_X_AXIS": LON_STATS.DEFAULT_X_AXIS, "DEFAULT_Y_AXIS": LON_STATS.DEFAULT_Y_AXIS,
        "PLOT_STYLE_OPTIONS": canon(LON_STATS.PLOT_STYLE_OPTIONS),
        "DEFAULT_PLOT_STYLE": LON_STATS.DEFAULT_PLOT_STYLE,
    }

    # graph statistics on the same input
    pairwise = GRAPH_STATS.compute_pairwise_correlations(NODE_STATS_INPUT)
    report["pairwise_correlations"] = canon(pairwise)
    report["correlation_pairs"] = {
        f"{x}|{y}": canon(GRAPH_STATS.compute_correlation_pair(NODE_STATS_INPUT, x, y))
        for x, y in axes
    }
    report["correlation_pair|single"] = canon(GRAPH_STATS.compute_correlation_pair(NODE_STATS_INPUT[:1], "neigh_feas", "error"))
    report["correlation_pair|constant"] = canon(GRAPH_STATS.compute_correlation_pair(
        [dict(s, neigh_feas=0.5) for s in NODE_STATS_INPUT], "neigh_feas", "error"))

    # the two Dash table builders, wherever they live
    report["table|correlations"] = safe_fig(TABLES.build_correlation_table, pairwise)
    report["table|correlations_empty"] = safe_fig(TABLES.build_correlation_table, [])
    selected = GRAPH_STATS.compute_correlation_pair(NODE_STATS_INPUT, "neigh_feas", "error")
    report["table|selected"] = safe_fig(
        TABLES.build_selected_correlation_display, selected, "Neighbourhood Feasibility", "Sampling Error")
    report["table|selected_none"] = safe_fig(
        TABLES.build_selected_correlation_display, {"pearson": None, "spearman": None, "n": 0}, "X", "Y")
    return report


# ------------------------------------------------------------------ 4. graph fixtures

def bits(value):
    """Deterministic 10-bit solution for an integer key."""
    return [int(c) for c in format(value, "010b")]


# Trajectory keys. Series A and B deliberately share solutions (0, 1, 3), so the posterior
# population path exercises its cross-series/run node sharing, and there are >30 unique
# solutions so t-SNE's default perplexity is satisfied.
TRAJECTORY_KEYS = [
    [[0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 1022],
     [0, 2, 6, 14, 46, 110, 238, 494]],
    [[1, 5, 21, 85, 341, 683, 687, 703, 767],
     [3, 11, 27, 59, 123, 251, 507, 1019]],
]
NOISY_XOR = 512

# Local optima: 32 nodes, of which 5 coincide with trajectory solutions (3, 15, 85, 341, 1023).
LON_KEYS = [3, 15, 85, 341, 1023] + [k for k in range(9, 1024, 33)][:27]
LON_EDGE_PAIRS = [(i, (i * 7 + 3) % len(LON_KEYS), 1 + (i % 5)) for i in range(len(LON_KEYS))] + [
    (i, (i + 1) % len(LON_KEYS), 2 + (i % 3)) for i in range(0, len(LON_KEYS), 3)]


def stn_entry(keys, base, with_alt=False, with_box=True, with_est=True):
    sols = [bits(k) for k in keys]
    noisy_sols = [bits(k ^ NOISY_XOR) for k in keys]
    fits = [float(base + 10 * i) for i in range(len(keys))]
    noisy_fits = [float(base + 10 * i + (1.5 if i % 2 else -2.25)) for i in range(len(keys))]
    iterations = [1 + i for i in range(len(keys))]
    evals = [10 + 5 * i for i in range(len(keys))]
    transitions = [[sols[i], sols[i + 1]] for i in range(len(sols) - 1)]
    box = [[f - 4.0, f - 2.0, f, f + 2.0, f + 4.0] for f in fits] if with_box else []
    est_adopted = [f + 0.5 for f in fits] if with_est else []
    est_discarded = [f - 0.75 for f in fits] if with_est else []
    cnt_adopted = [2 + i for i in range(len(keys))] if with_est else []
    cnt_discarded = [1 + i for i in range(len(keys))] if with_est else []
    alt_sols = [bits(k ^ 1) for k in keys] if with_alt else []
    alt_fits = [f - 5.0 for f in fits] if with_alt else []
    return [sols, fits, noisy_fits, iterations, transitions, [], box, noisy_sols,
            est_adopted, est_discarded, cnt_adopted, cnt_discarded, evals, alt_sols, alt_fits]


STN_SERIES = [
    [stn_entry(TRAJECTORY_KEYS[0][0], 100.0, with_alt=True),
     stn_entry(TRAJECTORY_KEYS[0][1], 104.0)],
    [stn_entry(TRAJECTORY_KEYS[1][0], 98.0),
     stn_entry(TRAJECTORY_KEYS[1][1], 102.0, with_alt=True)],
]
STN_LABELS = [["MuPlusLambda", 1.0], ["OnePlusOne", 2.0]]

CONTINUOUS_ENTRY = [
    [[0.5, -1.25], [0.75, -0.5], [1.5, 0.25]],
    [12.5, 9.75, 4.5],
    [13.0, 9.0, 5.5],
    [1, 2, 3],
    [[[0.5, -1.25], [0.75, -0.5]], [[0.75, -0.5], [1.5, 0.25]]],
    [], [], [[0.55, -1.2], [0.8, -0.45], [1.45, 0.3]],
    [], [], [], [], [10, 20, 30], [], [],
]


def lon_payload():
    optima = [bits(k) for k in LON_KEYS]
    fitness = [float(110 + (k % 47)) for k in LON_KEYS]
    key = lambda opt: ",".join(str(int(x)) for x in opt)
    return {
        "local_optima": [list(o) for o in optima],
        "fitness_values": fitness,
        "edge_transitions": [[list(optima[s]), list(optima[t])] for s, t, _ in LON_EDGE_PAIRS],
        "edge_weights": [w for _, _, w in LON_EDGE_PAIRS],
        "opt_feas_map": {key(o): (1 if i % 3 else 0) for i, o in enumerate(optima)},
        "neigh_feas_map": {key(o): round(0.03 * (i + 1), 3) for i, o in enumerate(optima)},
        "visit_prop_map": {key(o): round(0.02 * (i + 1), 3) for i, o in enumerate(optima)},
    }


def mo_front(gen, series, dual):
    front1 = [bits((gen * 13 + series * 101 + i * 7) % 1024) for i in range(3)]
    front2 = [bits((gen * 17 + series * 103 + i * 11) % 1024) for i in range(3)] if dual else None
    return {
        "front1": front1,
        "front2": front2,
        "metric1": float(10.0 + gen + series),
        "metric2": float(9.0 + gen + series) if dual else None,
        "gen_idx": gen,
    }


# Series 0 is dual-front (so the noisy MO nodes are embedded too) and long enough for t-SNE.
MO_DATA = [
    [[mo_front(g, 0, True) for g in range(20)]],
    [[mo_front(g, 1, False) for g in range(14)]],
]
MO_LABELS = [["NSGA2", 1.0], ["SEMO", 2.0]]


# ------------------------------------------------------------------ 5. the Dashboard orchestrator

# The three files are located by what they define, not by where they live, so the same contracts hold
# before and after the Stage 11 split. `layout/components.py` does not move.
LOGICAL_DASHBOARD = "dashboard/Dashboard.py"
LOGICAL_HELPERS = "dashboard/DashboardHelpers.py"
LAYOUT_COMPONENTS_PY = PKG / "dashboard" / "layout" / "components.py"

# Every module of the dashboard package except the layout package and the Stage-10 table builders.
def dashboard_modules():
    return [path for path in sorted((PKG / "dashboard").rglob("*.py"))
            if not module_key(path).startswith("dashboard/layout/")
            and module_key(path) != "dashboard/components.py"]


def module_defining(name):
    """The one dashboard module with a top-level `def name`."""
    found = [path for path in dashboard_modules()
             if any(isinstance(node, ast.FunctionDef) and node.name == name
                    for node in ast.parse(path.read_text()).body)]
    assert len(found) == 1, f"expected exactly one module defining {name}, found {found}"
    return found[0]


DASHBOARD_PY = module_defining("update_plot")
HELPERS_PY = module_defining("select_top_runs_by_fitness")

KEEP_DEFS = {
    "_add_guide_nodes", "_get_noise_param_label", "_get_so_xaxis_label", "_get_problem_goal",
    "_format_scientific", "_format_median_std", "_resolve_fit_column", "_resolve_evals_column",
    "_cap_noise", "_hide_series", "_build_stn_stats_table", "_filter_penalty",
}
KEEP_ASSIGNS = {"FIT_FUNC_XAXIS_LABELS", "FIT_FUNC_NOISE_PARAM_LABEL"}

SPY_NAMES = [
    "add_stn_trajectories", "add_mo_fronts", "add_prior_noise_stn_v4", "add_prior_noise_stn_v5",
    "add_prior_noise_stn_algo_pov", "add_lon_nodes", "add_lon_edges", "debug_mo_counts",
    "style_nodes", "calculate_lon_statistics", "_add_guide_nodes", "calculate_positions",
    "build_all_traces", "create_guide_traces", "create_axis_settings", "create_figure",
    "compute_node_feasibility_error", "compute_pairwise_correlations", "compute_correlation_pair",
    "plot_lon_stats", "plot_lon_stats_multi", "build_correlation_table",
    "build_selected_correlation_display", "parse_callback_inputs", "generate_run_summary_string",
]


def free_names(node):
    loads, stores, args, nested = set(), set(), set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            (loads if isinstance(child.ctx, ast.Load) else stores).add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            spec = child.args
            args.update(a.arg for a in spec.posonlyargs + spec.args + spec.kwonlyargs)
            if spec.vararg:
                args.add(spec.vararg.arg)
            if spec.kwarg:
                args.add(spec.kwarg.arg)
            if isinstance(child, ast.FunctionDef) and child is not node:
                nested.add(child.name)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            nested.update((a.asname or a.name).split(".")[0] for a in child.names)
        elif isinstance(child, ast.comprehension):
            for target in ast.walk(child.target):
                if isinstance(target, ast.Name):
                    nested.add(target.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            nested.add(child.name)
    return loads - stores - args - nested - set(dir(builtins))


def load_update_plot():
    """Execute the real Dashboard import block, helpers and `update_plot` without loading data."""
    source = DASHBOARD_PY.read_text()
    tree = ast.parse(source, filename=str(DASHBOARD_PY))

    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and not node.decorator_list and node.name in KEEP_DEFS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in KEEP_ASSIGNS:
            keep.append(node)

    namespace = {
        "__name__": package_of(DASHBOARD_PY) + "._probe",
        "__package__": package_of(DASHBOARD_PY),
        "__file__": str(DASHBOARD_PY),
        "__builtins__": builtins,
    }
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(DASHBOARD_PY), "exec"), namespace)

    callback = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "update_plot")
    stripped = copy.deepcopy(callback)
    stripped.decorator_list = []
    exec(compile(ast.Module(body=[stripped], type_ignores=[]), str(DASHBOARD_PY), "exec"), namespace)

    missing = sorted(n for n in free_names(callback) if n not in namespace and n not in ("Input", "Output", "app"))
    if missing:
        raise AssertionError(f"update_plot free names unresolved: {missing}")
    return namespace, [n.arg for n in callback.args.args]


def graph_snapshot(graph):
    """Ordered, un-canonicalised structure: node and edge iteration order is part of the contract."""
    nodes = [[name, canon(dict(data))] for name, data in graph.nodes(data=True)]
    edges = [[u, v, canon(key), canon(dict(data))] for u, v, key, data in graph.edges(keys=True, data=True)]
    return {
        "class": type(graph).__name__,
        "graph_attrs": canon(dict(graph.graph)),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_ids": digest([n for n, _ in nodes]),
        "nodes": digest(nodes),
        "edges": digest(edges),
        "attr_keys": sorted({k for _, data in graph.nodes(data=True) for k in data}),
        "edge_attr_keys": sorted({k for *_, data in graph.edges(keys=True, data=True) for k in data}),
        "types": sorted({str(data.get("type")) for _, data in graph.nodes(data=True)}),
        "edge_types": sorted({str(data.get("edge_type")) for *_, data in graph.edges(keys=True, data=True)}),
    }


def positions_snapshot(pos):
    if not isinstance(pos, dict):
        return {"class": type(pos).__name__, "repr": repr(pos)[:200]}
    items = [[node, canon(tuple(value))] for node, value in pos.items()]
    dims = sorted({len(value) for value in pos.values()}) if pos else []
    return {
        "class": type(pos).__name__,
        "count": len(pos),
        "keys": digest([node for node, _ in items]),
        "items": digest(items),
        "dims": dims,
        "all_finite": all(all(np.isfinite(float(c)) for c in value) for value in pos.values()),
    }


RESULT_RECORDED = {
    "add_lon_nodes", "add_stn_trajectories", "add_mo_fronts", "calculate_lon_statistics",
    "compute_node_feasibility_error", "compute_pairwise_correlations", "compute_correlation_pair",
}


def result_digest(value):
    try:
        return digest(value)
    except Exception:  # noqa: BLE001
        return digest(repr(value))


class Recorder:
    """Delegates to the real functions and records the call sequence and graph state (I-10b)."""

    def __init__(self):
        self.calls = []
        self.steps = []
        self.results = []
        self.before_layout = None
        self.positions = None

    def spy(self, name, real):
        def wrapper(*args, **kwargs):
            graph = args[0] if args and isinstance(args[0], nx.Graph) else None
            if name == "calculate_positions" and graph is not None:
                self.before_layout = graph_snapshot(graph)
            self.calls.append(name)
            result = real(*args, **kwargs)
            if graph is not None:
                snapshot = graph_snapshot(graph)
                self.steps.append([name, snapshot["nodes"], snapshot["edges"], snapshot["node_count"]])
            if name in RESULT_RECORDED:
                self.results.append([name, result_digest(result)])
            if name == "calculate_positions":
                self.positions = positions_snapshot(result)
            return result
        wrapper.__name__ = getattr(real, "__name__", name)
        return wrapper


BASE_KWARGS = dict(
    optimum=None, PID=KP_PID, opt_goal="maximise",
    options=["plot_3D", "LON_node_strength"],
    run_options=["noisy-nodes-square", "use-viridis", "curved-edges"],
    STN_lower_fit_limit=None, LO_fit_percent=100, LON_options=[],
    LON_node_colour_mode="fitness", LON_surface_colour="fitness", LON_edge_colour_feas=[],
    lmds_multiplier=1.0, NLON_fit_func="", NLON_intensity=1, NLON_samples=100, NLON_penalty=10,
    layout_value="mds", plot_type="RegLon", hover_info_value="fitness",
    azimuth_deg=35, elevation_deg=60,
    all_trajectories_list=None, STN_labels=None, run_start_index=0, n_runs_display=2,
    local_optima=None, axis_values={}, opacity_noise_bar=1, LON_node_opacity=1, LON_edge_opacity=1,
    STN_node_opacity=1, STN_edge_opacity=1, STN_node_min=5, STN_node_max=20,
    LON_node_min=10, LON_node_max=10.1, LON_edge_size_slider=5, STN_edge_size_slider=5,
    noisy_fitnesses_list=None, stn_plot_type="posterior", STN_MO_data=None,
    STN_MO_series_labels=None, stn_node_size_metric="evaluations",
    annotation_options=["annotate-start-nodes", "annotate-optimum", "annotate-end-nodes",
                        "annotate-mistakes", "annotate-info-panel"],
    fit_func="eval_noisy_kp_v1", info_panel_x=90, info_panel_y=75,
    axes_text_scale=1.0, annotation_text_scale=1.0, plot_theme="Viridis", plot_2d_data=None,
    lon_scatter_x="neigh_feas", lon_scatter_y="error", lon_scatter_plot_style="scatter",
    lon_scatter_multi_noise=[],
)

STN_ONLY = dict(all_trajectories_list=copy.deepcopy(STN_SERIES), STN_labels=copy.deepcopy(STN_LABELS))


def case_kwargs(**overrides):
    kwargs = copy.deepcopy(BASE_KWARGS)
    kwargs.update(copy.deepcopy(overrides))
    return kwargs


MIXED_CASES = {
    "M1": case_kwargs(
        **STN_ONLY, local_optima=lon_payload(), plot_type="NLon_box", NLON_fit_func="kpv1s",
        NLON_samples=25, optimum=150.0,
        run_options=["noisy-nodes-square", "use-viridis", "curved-edges", "show_alt_rep",
                     "show_alt_rep_no_fit", "show_noisy_path", "STN-hamming", "colour_by_evals",
                     "show_stn_boxplots", "show_estimated_adopted", "show_estimated_discarded"],
        LON_options=["LON-hamming", "LON-display-mesh", "LON-display-surface", "LON-node-diamond"],
        LON_node_colour_mode="neigh", LON_surface_colour="neigh_feas",
        annotation_options=["annotate-start-nodes", "annotate-optimum", "annotate-end-nodes",
                            "annotate-mistakes", "annotate-info-panel", "show-guides"],
        lon_scatter_multi_noise=["multi-noise"], NLON_intensity=3),
    "M2": case_kwargs(
        **STN_ONLY, local_optima=lon_payload(), plot_type="RegLon", stn_plot_type="prior_v4",
        layout_value="hamming_delta_ref", LON_node_colour_mode="feasible",
        LON_edge_colour_feas=["edge_feas"],
        run_options=["noisy-nodes-square", "use-viridis", "curved-edges", "show_alt_rep",
                     "use_est_discarded_as_base"],
        annotation_options=["annotate-start-nodes", "annotate-end-nodes", "annotate-mistakes",
                            "annotate-advanced-mistakes", "annotate-info-panel"]),
    "M3": case_kwargs(
        **STN_ONLY, local_optima=lon_payload(), plot_type="NLon_IQR", stn_plot_type="prior_v5",
        layout_value="lon_lmds", NLON_fit_func="kpv1s", NLON_samples=25, NLON_intensity=3,
        run_options=["noisy-nodes-square", "use-viridis", "curved-edges", "show_noisy_path"],
        lon_scatter_plot_style="violin", lon_scatter_multi_noise=["multi-noise"]),
    "M4": case_kwargs(
        **STN_ONLY, local_optima=lon_payload(), plot_type="RegLon", stn_plot_type="prior_algo_pov",
        NLON_fit_func="kpv1s", NLON_samples=25,
        annotation_options=["annotate-mistakes", "annotate-advanced-mistakes",
                            "annotate-start-nodes", "annotate-end-nodes"]),
    "M5": case_kwargs(
        **STN_ONLY, local_optima=lon_payload(), plot_type="NLon_box",
        stn_plot_type="posterior_algo_pov", layout_value="tsne", NLON_fit_func="kpv1s",
        NLON_samples=25, LON_options=["LON-visit-size"],
        run_options=["use-viridis", "curved-edges"]),
    "M6": case_kwargs(
        local_optima=lon_payload(), plot_type="RegLon", stn_plot_type="multiobjective",
        STN_MO_data=copy.deepcopy(MO_DATA), STN_MO_series_labels=copy.deepcopy(MO_LABELS),
        NLON_fit_func="kpv1s", NLON_samples=25, n_runs_display=1,
        annotation_options=["annotate-start-nodes", "annotate-optimum", "annotate-info-panel",
                            "show-guides"]),
    "M7": case_kwargs(
        local_optima=lon_payload(), plot_type="NLon_box", NLON_fit_func="kpv1s", NLON_samples=25,
        LON_node_colour_mode="feasible"),
    "M8": case_kwargs(**STN_ONLY, plot_type="RegLon"),
    "M9": case_kwargs(plot_type="RegLon"),
    "M10": case_kwargs(all_trajectories_list=[[copy.deepcopy(CONTINUOUS_ENTRY)]],
                       STN_labels=[["CMAES", 1.0]], plot_type="RegLon",
                       annotation_options=["annotate-start-nodes", "annotate-end-nodes"]),
}


def mixed_report(namespace, param_names):
    report = {}
    real = {name: namespace[name] for name in SPY_NAMES if name in namespace}
    update_plot = namespace["update_plot"]

    for case, kwargs in MIXED_CASES.items():
        recorder = Recorder()
        for name, fn in real.items():
            namespace[name] = recorder.spy(name, fn)
        random.seed(0)
        np.random.seed(0)
        buffer = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(buffer):
                outputs = update_plot(**copy.deepcopy(kwargs))
        except Exception as exc:  # noqa: BLE001
            outputs, error = None, f"{type(exc).__name__}: {exc}"
        finally:
            for name, fn in real.items():
                namespace[name] = fn

        entry = {
            "calls": recorder.calls,
            "steps": recorder.steps,
            "results": recorder.results,
            "before_layout": recorder.before_layout,
            "positions": recorder.positions,
            "error": error,
            "stdout": digest(buffer.getvalue().splitlines()),
            "stdout_lines": len(buffer.getvalue().splitlines()),
            "diagnostic": "ERROR: No solutions for Positioning" in buffer.getvalue(),
        }
        if outputs is not None:
            entry["outputs"] = [fig_digest(o) if hasattr(o, "to_plotly_json") else digest(o) for o in outputs]
            entry["output_types"] = [type(o).__name__ for o in outputs]
            entry["figure_semantic"] = advanced_sorted_digest(outputs[0])
            entry["advanced_traces"] = sorted(
                (t.get("name") or "") for t in outputs[0].to_plotly_json().get("data", [])
                if isinstance(t.get("name"), str) and t["name"].startswith(ADVANCED_TRACE_PREFIX))
        report[case] = entry
    return report


# ------------------------------------------------------------------ 6. layout smoke matrix

LAYOUTS = ["spring", "kamada_kawai", "mds", "r_lmds", "fps_lmds", "lon_lmds", "tsne", "raw",
           "hamming_delta_ref"]
NONDETERMINISTIC = {("spring", "so"), ("fps_lmds", "so")}


def build_layout_fixtures(namespace):
    """Graphs populated through the real population functions, for the layout matrix."""
    parse = namespace["parse_callback_inputs"]
    add_stn = namespace["add_stn_trajectories"]
    add_lon_nodes = namespace["add_lon_nodes"]
    add_lon_edges = namespace["add_lon_edges"]
    add_mo = namespace["add_mo_fronts"]

    def config_for(**overrides):
        kwargs = copy.deepcopy(BASE_KWARGS)
        kwargs.update(overrides)
        accepted = inspect.signature(parse).parameters
        mapped = dict(
            optimum=kwargs["optimum"], pid=kwargs["PID"], opt_goal=kwargs["opt_goal"],
            options=kwargs["options"], run_options=kwargs["run_options"],
            stn_lower_fit_limit=kwargs["STN_lower_fit_limit"], lo_fit_percent=kwargs["LO_fit_percent"],
            lon_options=kwargs["LON_options"], lon_node_colour_mode=kwargs["LON_node_colour_mode"],
            lon_surface_colour=kwargs["LON_surface_colour"],
            lon_edge_colour_feas=kwargs["LON_edge_colour_feas"], lmds_multiplier=kwargs["lmds_multiplier"],
            nlon_fit_func=kwargs["NLON_fit_func"], nlon_intensity=kwargs["NLON_intensity"],
            nlon_samples=kwargs["NLON_samples"], nlon_penalty=kwargs["NLON_penalty"],
            layout_value=kwargs["layout_value"], plot_type=kwargs["plot_type"],
            hover_info_value=kwargs["hover_info_value"], azimuth_deg=kwargs["azimuth_deg"],
            elevation_deg=kwargs["elevation_deg"], run_start_index=kwargs["run_start_index"],
            n_runs_display=kwargs["n_runs_display"], axis_values=kwargs["axis_values"],
            opacity_noise_bar=kwargs["opacity_noise_bar"], lon_node_opacity=kwargs["LON_node_opacity"],
            lon_edge_opacity=kwargs["LON_edge_opacity"], stn_node_opacity=kwargs["STN_node_opacity"],
            stn_edge_opacity=kwargs["STN_edge_opacity"], stn_node_min=kwargs["STN_node_min"],
            stn_node_max=kwargs["STN_node_max"], lon_node_min=kwargs["LON_node_min"],
            lon_node_max=kwargs["LON_node_max"], lon_edge_size_slider=kwargs["LON_edge_size_slider"],
            stn_edge_size_slider=kwargs["STN_edge_size_slider"], stn_plot_type=kwargs["stn_plot_type"],
            node_size_metric=kwargs["stn_node_size_metric"], colorscale=kwargs["plot_theme"],
        )
        return parse(**{k: v for k, v in mapped.items() if k in accepted})

    convert = namespace["convert_to_single_edges_format"]
    filter_lo = namespace["filter_local_optima"]

    def populate_stn(graph, config):
        mapping = {}
        for idx, runs in enumerate(copy.deepcopy(STN_SERIES)):
            mapping = add_stn(graph, runs[:2], config.algo_colors[idx], idx, mapping, config)

    def populate_lon(graph, config):
        processed = filter_lo(convert(lon_payload()), config.lon.fit_percent)
        mapping, _ = add_lon_nodes(graph, processed, {}, config, KP_PID)
        add_lon_edges(graph, processed, mapping, config, None)

    fixtures = {}

    config_so = config_for()
    graph = nx.MultiDiGraph()
    populate_stn(graph, config_so)
    fixtures["stn_only"] = (graph, "posterior")

    graph = nx.MultiDiGraph()
    populate_lon(graph, config_so)
    fixtures["lon_only"] = (graph, "posterior")

    graph = nx.MultiDiGraph()
    populate_stn(graph, config_so)
    populate_lon(graph, config_so)
    fixtures["mixed"] = (graph, "posterior")

    config_mo = config_for(stn_plot_type="multiobjective")
    graph = nx.MultiDiGraph()
    for idx, runs in enumerate(copy.deepcopy(MO_DATA)):
        add_mo(graph, runs, config_mo.algo_colors[idx], idx, config_mo.noisy_node_color)
    fixtures["mo"] = (graph, "multiobjective")
    return fixtures


def layout_report(namespace):
    calculate_positions = namespace["calculate_positions"]
    fixtures = build_layout_fixtures(namespace)
    report = {}

    for fixture_name, (graph, stn_plot_type) in fixtures.items():
        kind = "mo" if fixture_name == "mo" else "so"
        for layout in LAYOUTS:
            label = f"{layout}|{fixture_name}"
            smoke = (layout, kind) in NONDETERMINISTIC
            runs = []
            for seed in ((0, 12345) if smoke else (0,)):
                random.seed(seed)
                np.random.seed(seed)
                buffer = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buffer):
                        pos = calculate_positions(graph, layout, stn_plot_type, True, 1.0)
                    runs.append({"error": None, **positions_snapshot(pos)})
                except Exception as exc:  # noqa: BLE001
                    runs.append({"error": f"{type(exc).__name__}", "message": str(exc)[:200]})
            entry = {"smoke_only": smoke, "error": runs[0]["error"]}
            if smoke:
                entry["structural"] = [
                    {k: run.get(k) for k in ("error", "class", "count", "keys", "dims", "all_finite")}
                    for run in runs
                ]
            else:
                entry["exact"] = runs[0]
            report[label] = entry
    return report


# ------------------------------------------------------------------ 7. dashboard bodies and imports

VIZ_MODULE_PREFIXES = ("noisyvis.visualization", "noisyvis.plotting", "noisyvis.viz",
                       "noisyvis.analysis", "noisyvis.dashboard.components")


def body_hash(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return {"ast": sha256(ast.dump(tree)), "literals": literal_digest(tree),
            "statements": len(tree.body)}


def resolve_module(node, package):
    if node.level == 0:
        return node.module
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join([*base, node.module] if node.module else base)


def import_sources():
    """Which modules carry each logical file's imports, before and after the Stage 11 split.

    Stage 11 spreads Dashboard.py's imports over the modules that replace it, so the names are
    aggregated over that group; the values they resolve to are unchanged (amendment A1 keeps every
    existing explicit import, so the pinned IMPORTS surface does not shrink).
    """
    helpers = HELPERS_PY
    dashboard_group = [path for path in dashboard_modules() if path != helpers]
    return {
        LOGICAL_DASHBOARD: dashboard_group,
        LOGICAL_HELPERS: [helpers],
        module_key(LAYOUT_COMPONENTS_PY): [LAYOUT_COMPONENTS_PY],
    }


def import_report():
    report = {}
    for logical_key, paths in import_sources().items():
        names, modules = [], []
        for path in paths:
            package = package_of(path)
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = resolve_module(node, package)
                if module and module.startswith(VIZ_MODULE_PREFIXES):
                    modules.append(module)
                    names.extend(alias.name for alias in node.names)
        names = sorted(set(names))
        resolved = {}
        for name in sorted(set(names)):
            owner = None
            for module in modules:
                candidate = importlib.import_module(module)
                if hasattr(candidate, name):
                    owner = candidate
                    break
            value = getattr(owner, name, None) if owner else None
            if value is None:
                resolved[name] = None
            elif callable(value) and not isinstance(value, type):
                resolved[name] = {"kind": "callable", "name": value.__name__, "ast": obj_ast_sha256(value)}
            elif isinstance(value, type):
                resolved[name] = {"kind": "class", "name": value.__name__, "ast": obj_ast_sha256(value)}
            else:
                resolved[name] = {"kind": "value", "canon": canon(value)}
        report[logical_key] = {
            "names": sorted(names), "name_count": len(names),
            "modules": sorted(set(modules)), "resolved": resolved,
        }
    return report


# ------------------------------------------------------------------ main

SECTIONS = __SECTIONS__


def main():
    import time
    timings = {}

    def timed(name, fn, *args):
        start = time.time()
        value = fn(*args)
        timings[name] = round(time.time() - start, 2)
        return value

    report = {
        "meta": {
            "python": sys.version.split()[0],
            "source_root": str(SOURCE_ROOT),
            "noisyvis_file": importlib.import_module("noisyvis").__file__,
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "facades": {
                "viz": VIZ.__name__, "plots": PLOTS.__name__, "graph_stats": GRAPH_STATS.__name__,
                "lon_stats": LON_STATS.__name__, "tables": TABLES.__name__,
            },
        },
    }

    if "core" in SECTIONS:
        definitions, duplicates, monoliths = timed("definitions", scan_definitions)
        report["definitions"] = definitions
        report["duplicates"] = duplicates
        report["monoliths"] = monoliths
        report["registry"] = timed("registry", registry_report)
        report["performance"] = timed("performance", performance_report)
        report["lon_stats"] = timed("lon_stats", lon_stats_report)
        # Dashboard.py and DashboardHelpers.py are rebuilt from wherever their statements now live,
        # in their frozen PRE order, with the Stage-11 transformations inverted; the hashes below are
        # therefore still the Stage-10 values, computed over the same body.
        constants = INV.store_constants(PKG)
        report["bodies"] = {
            LOGICAL_DASHBOARD: INV.reassemble(PKG, LOGICAL_DASHBOARD,
                                              BODY_KEYS[LOGICAL_DASHBOARD], constants),
            LOGICAL_HELPERS: INV.reassemble(PKG, LOGICAL_HELPERS,
                                            BODY_KEYS[LOGICAL_HELPERS], constants),
            module_key(LAYOUT_COMPONENTS_PY): body_hash(LAYOUT_COMPONENTS_PY),
        }
        report["imports"] = timed("imports", import_report)
        report["obsolete_import_references"] = obsolete_import_references()

    if "pareto" in SECTIONS:
        report["pareto"] = timed("pareto", pareto_report)

    if "heavy" in SECTIONS:
        namespace, param_names = timed("load_update_plot", load_update_plot)
        report["meta"]["update_plot_params"] = param_names
        report["mixed"] = timed("mixed", mixed_report, namespace, param_names)
        report["layouts"] = timed("layouts", layout_report, namespace)

    report["meta"]["timings"] = timings
    report["meta"]["sections"] = SECTIONS
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))


main()
'''


def build_probe(source_root, sections) -> str:
    """The probe for one source tree and one group of sections."""
    replacements = {
        "__SOURCE_ROOT__": repr(str(source_root)),
        "__WORKSPACE__": repr(str(WORKSPACE)),
        "__FENCE__": repr(str(HARNESS_DIR / "fence.py")),
        "__INVENTORY__": repr(str(HARNESS_DIR / "dashboard_inventory.py")),
        "__BODY_KEYS__": repr(BODY_KEYS),
        "__SECTIONS__": repr(list(sections)),
    }
    probe = _PROBE
    for marker, value in replacements.items():
        probe = probe.replace(marker, value)
    return probe


def _run_probe(sections) -> dict:
    """Run one probe subprocess in a fresh temp root with the write fence installed."""
    root = make_temp_root()
    env = child_env(root)
    # Pinned for this subprocess only; see the module docstring. No production code or container
    # environment is affected.
    env["PYTHONHASHSEED"] = "0"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", build_probe(WORKSPACE, sections)],
            cwd=str(root), env=env, capture_output=True, text=True, timeout=1800,
        )
        assert completed.returncode == 0, (
            f"viz probe {sections} failed:\n{completed.stderr[-4000:]}")
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert report["meta"]["noisyvis_file"].startswith(str(WORKSPACE / "src" / "noisyvis")), report["meta"]
    assert report["meta"]["hash_seed"] == "0", report["meta"]
    return report


@pytest.fixture(scope="module")
def probe() -> dict:
    """Cheap contracts: definitions, registry, performance, LON stats, bodies and imports."""
    return _run_probe(["core"])


@pytest.fixture(scope="module")
def pareto_probe() -> dict:
    """Every Pareto plot, including the MDS/t-SNE/Isomap distance variants."""
    return _run_probe(["pareto"])


@pytest.fixture(scope="module")
def heavy_probe() -> dict:
    """The real update_plot over M1-M10 and the nine-layout smoke matrix."""
    return _run_probe(["heavy"])


def _import_fails(module: str) -> str:
    """Import `module` in a fresh interpreter; return the exception class name, or "IMPORTED".

    A separate process, so nothing already in this process's sys.modules can mask the result.
    """
    lines = [
        "import importlib",
        "try:",
        f"    importlib.import_module({module!r})",
        "    print('IMPORTED')",
        "except Exception as exc:",
        "    print(type(exc).__name__)",
    ]
    completed = subprocess.run([sys.executable, "-c", chr(10).join(lines)],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return completed.stdout.strip()


def _mismatches(observed: dict, expected: dict) -> list:
    keys = sorted(set(observed) | set(expected))
    return [f"{key}:\n    expected {expected.get(key)!r}\n    observed {observed.get(key)!r}"
            for key in keys if observed.get(key) != expected.get(key)]


# ------------------------------------------------------------------------------ the tests


def test_viz_definitions_pinned_and_unique(probe):
    observed = probe["definitions"]

    assert set(observed) == set(DEFINITIONS), (
        "the visualisation/plotting definition set changed:\n"
        f"  missing: {sorted(set(DEFINITIONS) - set(observed))}\n"
        f"  added:   {sorted(set(observed) - set(DEFINITIONS))}"
    )

    for name in sorted(DEFINITIONS):
        record = dict(observed[name])
        module = record.pop("module")
        assert record == DEFINITIONS[name], (
            f"{name}: definition changed (normalised AST or string literals):\n  "
            + "\n  ".join(_mismatches(record, DEFINITIONS[name]))
        )
        assert module in allowed_modules(name), (
            f"{name} is defined in {module}, which is not one of {allowed_modules(name)}"
        )

    # Only `_viridis_colors` may exist twice, once per performance module, and only at an approved
    # pre- or post-move location. The count, the basenames and the locations are all pinned.
    duplicates = probe["duplicates"]
    assert set(duplicates) == set(DUPLICATE_DEFINITIONS), (
        f"the set of duplicated definitions changed: {sorted(duplicates)}"
    )
    for name, modules in sorted(duplicates.items()):
        assert len(modules) == DUPLICATE_DEFINITIONS[name], f"{name}: {modules}"
        assert len(set(modules)) == DUPLICATE_DEFINITIONS[name], f"{name}: repeated module {modules}"
        assert all(module in ALLOWED_DUPLICATE_MODULES[name] for module in modules), (
            f"{name} is defined at an unapproved location: {modules}"
        )
        assert sorted(module.rsplit("/", 1)[1] for module in modules) == DUPLICATE_BASENAMES[name], modules

    # The legacy monolith is the only module excluded from the inventory, and only while it exists.
    monoliths = probe["monoliths"]
    assert monoliths == [], f"Checkpoint F removed the legacy monolith, but found: {monoliths}"
    assert ALLOWED_MONOLITHS == (), ALLOWED_MONOLITHS

    # Checkpoint E: the pre-move packages are gone. No source file may import them, and neither may
    # be importable at all. The import check runs in its own fresh subprocess so that nothing already
    # in this process's sys.modules can mask the result.
    assert probe["obsolete_import_references"] == [], (
        "source still imports a pre-move package: "
        + "; ".join(probe["obsolete_import_references"])
    )
    for package in OBSOLETE_PACKAGES:
        assert _import_fails(package) == "ModuleNotFoundError", (
            f"{package} is still importable: {_import_fails(package)}"
        )

    # The graph-population split keeps every definition on exactly one side.
    for name in STN_POPULATION:
        assert observed[name]["module"] in (DEFINITION_MODULES[name], "viz/graph/stn.py"), name
    for name in LON_POPULATION:
        assert observed[name]["module"] in (DEFINITION_MODULES[name], "viz/graph/lon.py"), name


def test_plot_registry_pinned(probe):
    observed = probe["registry"]

    assert observed["pareto_keys"] == REGISTRY["pareto_keys"]
    assert observed["performance_keys"] == REGISTRY["performance_keys"]
    assert observed["dropdown_values"] == REGISTRY["pareto_keys"], (
        "the Dashboard Pareto dropdown no longer matches the registry keys"
    )
    for family in ("pareto", "performance"):
        assert not _mismatches(observed[family], REGISTRY[family]), (
            f"{family} registry changed:\n  " + "\n  ".join(_mismatches(observed[family], REGISTRY[family]))
        )
        assert all(entry["lookup_is_same"] for entry in observed[family].values())
    assert observed["unknown_key_is_none"] is True
    # The 9 plot2d_* performance aliases are live and must keep pointing at their targets.
    assert all(observed["performance_aliases"].values()), observed["performance_aliases"]

    # Checkpoint F: the 12 camelCase Pareto aliases and the legacy monolith are gone. Absence is
    # checked positively — in both modules' __dict__ and both modules' __all__ — so that nothing can
    # pass vacuously.
    assert observed["removed_aliases_present"] == {}, (
        "removed Pareto aliases are still exposed: " + repr(observed["removed_aliases_present"])
    )
    assert observed["alias_names_present"] == [], observed["alias_names_present"]
    assert sorted(observed["canonical_targets_present"]) == sorted(set(REMOVED_PARETO_ALIASES.values())), (
        "a canonical Pareto function disappeared with its alias: "
        + repr(observed["canonical_targets_present"])
    )
    assert observed["monolith_present"] is False, "the legacy Pareto monolith is still present"


def test_pareto_figures_pinned(pareto_probe):
    observed = pareto_probe["pareto"]

    assert set(observed) == set(PARETO_FIGURES), (
        f"  missing: {sorted(set(PARETO_FIGURES) - set(observed))}\n"
        f"  added:   {sorted(set(observed) - set(PARETO_FIGURES))}"
    )
    changed = _mismatches(observed, PARETO_FIGURES)
    assert not changed, "Pareto figures changed:\n  " + "\n  ".join(changed)


def test_performance_figures_pinned(probe):
    observed = probe["performance"]

    changed = _mismatches(observed, PERFORMANCE_FIGURES)
    assert not changed, "performance figures changed:\n  " + "\n  ".join(changed)


def test_lon_stats_figures_and_tables_pinned(probe):
    observed = probe["lon_stats"]

    changed = _mismatches(observed, LON_STATS)
    assert not changed, "LON-stats figures, statistics or Dash tables changed:\n  " + "\n  ".join(changed)


# ------------------------------------------------------------------ the shared-graph orchestration

POPULATION_STN_CALLS = frozenset({
    "add_stn_trajectories", "add_mo_fronts", "add_prior_noise_stn_v4", "add_prior_noise_stn_v5",
    "add_prior_noise_stn_algo_pov",
})


def _indices(calls, *names):
    wanted = frozenset(names)
    return [index for index, call in enumerate(calls) if call in wanted]


def _assert_chronology(case, calls):
    """Invariant I-10b, proven from the recorded call sequence rather than from a digest alone."""
    stn = _indices(calls, *POPULATION_STN_CALLS)
    lon_nodes = _indices(calls, "add_lon_nodes")
    lon_edges = _indices(calls, "add_lon_edges")
    styling = _indices(calls, "style_nodes")
    statistics = _indices(calls, "calculate_lon_statistics")
    guides = _indices(calls, "_add_guide_nodes")
    layout = _indices(calls, "calculate_positions")
    traces = _indices(calls, "build_all_traces")
    figure = _indices(calls, "create_figure")

    assert len(layout) == 1, f"{case}: expected exactly one positioning call, got {len(layout)}"
    assert len(styling) == 1 and len(statistics) == 1, case

    if stn and lon_nodes:
        assert max(stn) < min(lon_nodes), f"{case}: LON population must follow STN population"
    if lon_nodes and lon_edges:
        assert min(lon_nodes) < lon_edges[0], f"{case}: LON nodes must precede LON edges"
        assert lon_edges[0] < styling[0], f"{case}: styling must follow LON edge population"
    if stn:
        assert max(stn) < styling[0], f"{case}: styling must follow STN population"
    assert styling[0] < statistics[0] < layout[0], f"{case}: styling -> statistics -> positioning"
    if guides:
        assert statistics[0] < guides[0] < layout[0], f"{case}: guides land after statistics, before layout"
    if traces:
        assert layout[0] < traces[0], f"{case}: traces must follow positioning"
    if figure:
        # Multi-noise LON resampling happens after the main figure, and only there.
        late = [index for index in lon_nodes if index > figure[0]]
        assert late == MIXED[case]["late_lon_population"], (
            f"{case}: multi-noise LON repopulation moved: {late}"
        )


def test_update_plot_mixed_pipeline_pinned(heavy_probe):
    mixed = heavy_probe["mixed"]

    assert sorted(mixed) == sorted(MIXED), sorted(mixed)

    for case in sorted(MIXED):
        observed, expected = mixed[case], MIXED[case]
        assert observed["error"] == expected["error"], f"{case}: {observed['error']}"
        _assert_chronology(case, observed["calls"])

        for field in ("calls", "steps", "results", "before_layout", "positions", "outputs",
                      "output_types", "figure_semantic", "advanced_traces", "stdout",
                      "stdout_lines", "diagnostic"):
            assert observed.get(field) == expected.get(field), (
                f"{case}: {field} changed\n    expected {expected.get(field)!r}"
                f"\n    observed {observed.get(field)!r}"
            )

    # M6: the MO population positions only its own nodes, so the LON stays unpositioned and unrendered.
    assert mixed["M6"]["before_layout"]["node_count"] > mixed["M6"]["positions"]["count"]
    assert "LON" in mixed["M6"]["before_layout"]["types"]
    # M9: the empty graph keeps its diagnostic.
    assert mixed["M9"]["diagnostic"] is True
    assert mixed["M9"]["positions"]["count"] == 0
    # M7 and M8 are the independent-population controls.
    assert mixed["M7"]["before_layout"]["types"] == ["LON"]
    assert "LON" not in mixed["M8"]["before_layout"]["types"]

    # ---- layout smoke matrix: nine Dashboard layouts over four graph populations
    layouts = heavy_probe["layouts"]
    assert sorted(layouts) == sorted(LAYOUTS), sorted(layouts)

    for label in sorted(LAYOUTS):
        observed, expected = layouts[label], LAYOUTS[label]
        assert observed["smoke_only"] == expected["smoke_only"], label
        assert observed["error"] == expected["error"], f"{label}: {observed}"
        if expected["smoke_only"]:
            # Nondeterministic embeddings: structural contract only, under two RNG states.
            for run in observed["structural"]:
                assert run["error"] is None, f"{label}: {run}"
                assert run["class"] == "dict"
                assert run["all_finite"] is True, f"{label}: non-finite coordinate"
            assert observed["structural"] == expected["structural"], (
                f"{label}: structural contract changed\n    expected {expected['structural']}"
                f"\n    observed {observed['structural']}"
            )
        else:
            assert observed["exact"] == expected["exact"], (
                f"{label}: positions changed\n    expected {expected['exact']}"
                f"\n    observed {observed['exact']}"
            )


def test_dashboard_bodies_unchanged(probe):
    observed = probe["bodies"]

    changed = _mismatches(observed, BODIES)
    assert not changed, (
        "a Dashboard-side body changed; Stage 10 may only change import statements in these files:\n  "
        + "\n  ".join(changed)
    )


def test_dashboard_viz_imports_resolve(probe):
    observed = probe["imports"]

    assert sorted(observed) == sorted(IMPORTS), sorted(observed)
    for path in sorted(IMPORTS):
        expected = IMPORTS[path]
        assert observed[path]["names"] == expected["names"], (
            f"{path}: imported visualisation names changed:\n"
            f"  missing: {sorted(set(expected['names']) - set(observed[path]['names']))}\n"
            f"  added:   {sorted(set(observed[path]['names']) - set(expected['names']))}"
        )
        assert observed[path]["name_count"] == expected["name_count"], path
        changed = _mismatches(observed[path]["resolved"], expected["resolved"])
        assert not changed, f"{path}: imported objects changed:\n  " + "\n  ".join(changed)
