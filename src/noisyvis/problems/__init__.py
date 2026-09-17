"""Problem definitions by family (plan §5.7).

The runners resolve each config's `problem.fitness_fn` with
getattr(sys.modules['noisyvis.problems'], name), so every evaluator is re-exported here explicitly,
together with the instance-loader and constraint APIs.
"""

from .onemax import (
    OneMax_fitness,
    OneMax_prior_bitflip_fitness,
    OneMax_prior_mult_bitflip_fitness,
    OneMax_prior_pq_bitwise_fitness,
    OneMax_prior_1q_bitwise_fitness,
)
from .jump import jump_fitness
from .knapsack import (
    eval_ind_kp,
    eval_noisy_kp_v1_simple,
    eval_noisy_kp_v2_simple,
    eval_noisy_kp_v1,
    eval_noisy_kp_v1_penalty,
    eval_noisy_kp_v2,
    eval_noisy_kp_v2_penalty,
    eval_noisy_kp_v3,
    eval_noisy_kp_prior_bitflip,
    eval_noisy_kp_prior_mult_bitflip,
    eval_noisy_kp_pq_prior_bitwise,
    eval_noisy_kp_1q_prior_bitwise,
)
from .continuous import rastrigin_eval, birastrigin_eval, ackley
from .instances import load_problem_KP, get_knapsack_problem_stats, interpret_correlation
from .knapsack_mo import eval_noisy_kp_v1_mo, eval_noisy_kp_v1_mo_violation, countingOnesCountingZeros
from .constraints import knap_violation
