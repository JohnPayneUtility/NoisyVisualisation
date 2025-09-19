from deap import base, creator, tools
from .LONs import random_bit_flip

def BinaryCoLON(pert_attempts, len_sol, weights,
              attr_function=None,
              n_flips_mut=1,
              n_flips_pert=2,
              mutate_function=None,
              perturb_function=None,
              improv_method='best',
              fitness_function=None,
              violation_function=None,   # <-- NEW: (func, kwargs) returning scalar v(x)
              starting_solution=None,
              true_fitness_function=None,
              target_stop=None):
    """
    LON construction using Deb's constraint-handling preorder for comparisons.
    Debs rule (a ⊲ b):
      1) If v(a) <= 0 and v(b) > 0       -> a better
      2) If v(a) > 0 and v(b) > 0        -> lower violation wins
      3) If v(a) <= 0 and v(b) <= 0      -> higher objective f wins
    """

    if violation_function is None:
        raise ValueError("violation_function is required for Deb's preorder (returns scalar v(x); <=0 means feasible).")

    # 1) Create Fitness and Individual classes if not existing
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)

    # 2) Helper: generate all n_flips bit flips
    def generate_bit_flip_combinations(ind, n_flips):
        import itertools
        indices = range(len(ind))
        combinations = itertools.combinations(indices, n_flips)
        mutants = []
        for combo in combinations:
            mutant = toolbox.clone(ind)
            for index in combo:
                mutant[index] = 1 - mutant[index]
            mutants.append(mutant)
        return mutants

    # ---- Deb comparator ----
    def is_feasible(ind):
        return getattr(ind, "violation", None) is not None and ind.violation <= 0.0

    def deb_better(a, b):
        # optional eps if you want numerical tolerance:
        # eps = 0.0
        va, vb = a.violation, b.violation
        fa = a.fitness.values[0]
        fb = b.fitness.values[0]

        # 1) feasible beats infeasible
        if va <= 0.0 and vb > 0.0:
            return True
        if va > 0.0 and vb <= 0.0:
            return False

        # 2) both infeasible -> lower violation is better
        if va > 0.0 and vb > 0.0:
            return va < vb

        # 3) both feasible -> higher objective is better
        # if abs(fa - fb) <= eps: return False
        return fa > fb

    # ---- Evaluators (objective & violation) ----
    def evaluate_objective(ind):
        # expects fitness_function as (callable, kwargs) returning a tuple or a scalar
        val = fitness_function[0](ind, **fitness_function[1])
        # Normalize to tuple for DEAP Fitness
        return val if isinstance(val, tuple) else (val,)

    def evaluate_violation(ind):
        # expects violation_function as (callable, kwargs) returning scalar v(x)
        v = violation_function[0](ind, **violation_function[1])
        return float(v)

    def evaluate_both(ind):
        ind.fitness.values = evaluate_objective(ind)
        ind.violation = evaluate_violation(ind)

    # 3) Toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_objective)
    toolbox.register("violate", evaluate_violation)
    toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))
    toolbox.register("perturb", lambda ind: perturb_function[0](ind, **perturb_function[1]))

    # 4) Create and evaluate a starting solution
    if starting_solution is not None:
        individual = creator.Individual(starting_solution)
    else:
        individual = creator.Individual([toolbox.attribute() for _ in range(len_sol)])
    evaluate_both(individual)

    # 5) Data
    local_optima = []
    fitness_values = []   # still record objective f at attractor
    edges = {}
    prev_local_opt = None
    prev_fitness = None

    # 6) Basin-hopping
    pert_attempt = 0

    while pert_attempt < pert_attempts:
        if improv_method == 'best':
            # Best-improvement local search under Deb’s preorder
            improvement = True
            while improvement:
                mutants = generate_bit_flip_combinations(individual, n_flips_mut)

                # Evaluate neighbours (both f and v)
                for m in mutants:
                    if hasattr(m.fitness, "values"):
                        del m.fitness.values
                    m.fitness.values = toolbox.evaluate(m)
                    m.violation = toolbox.violate(m)

                # Pick the best among neighbours + current (Deb’s preorder)
                candidates = mutants + [individual]
                best_mutant = candidates[0]
                for cand in candidates[1:]:
                    if deb_better(cand, best_mutant):
                        best_mutant = cand

                # Move only if strictly better under Deb
                if deb_better(best_mutant, individual):
                    individual[:] = best_mutant
                    # recompute f and v for the moved-to individual (safe after copy)
                    if hasattr(individual.fitness, "values"):
                        del individual.fitness.values
                    evaluate_both(individual)
                else:
                    improvement = False

            # Reached a local (Deb) optimum
            current_local_opt = tuple(individual)
            current_fitness = individual.fitness.values[0]

            if prev_fitness is not None and current_fitness < prev_fitness and is_feasible(individual):
                print("Alert: Non-monotonic transition (objective dropped but Deb may still allow via violation). "
                      f"Prev f: {prev_fitness}, Curr f: {current_fitness}")

            if current_local_opt not in local_optima:
                local_optima.append(current_local_opt)
                fitness_values.append(current_fitness)

            if prev_local_opt is not None and prev_local_opt != current_local_opt:
                edges[(prev_local_opt, current_local_opt)] = edges.get((prev_local_opt, current_local_opt), 0) + 1

            prev_local_opt = current_local_opt
            prev_fitness = current_fitness

            if target_stop is not None and current_fitness >= target_stop and is_feasible(individual):
                # Only early-stop if target achieved feasibly (usual practice)
                import logging
                logging.info(f"Target fitness reached feasibly: {current_fitness} >= {target_stop}")
                break

            # Perturbation (escape) step
            pert_attempt += 1
            perturbed = toolbox.clone(individual)
            perturbed[:], _ = random_bit_flip(perturbed, n_flips=n_flips_pert, exclude_indices=None)
            if hasattr(perturbed.fitness, "values"):
                del perturbed.fitness.values
            evaluate_both(perturbed)

            if pert_attempt % 100 == 0:
                import logging
                logging.info(f"PertAttmpt: {pert_attempt}")

            # Accept perturbed if better by Deb’s rule; if so, reset attempts
            if deb_better(perturbed, individual):
                individual[:] = perturbed
                if hasattr(individual.fitness, "values"):
                    del individual.fitness.values
                evaluate_both(individual)
                pert_attempt = 0
                import logging
                logging.info(f"Accepted perturbation via Deb's rule. Curr f={individual.fitness.values[0]:.6f}, "
                             f"v={individual.violation:.6g}")

        elif improv_method == 'first':
            raise NotImplementedError("first-improvement not implemented yet.")

    # 7) Pack edges
    edges_list = [(source, target, weight) for (source, target), weight in edges.items()]

    return local_optima, fitness_values, edges_list
