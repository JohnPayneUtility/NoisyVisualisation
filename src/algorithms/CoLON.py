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
                violation_function=None,   # (func, kwargs) returning scalar v(x); <=0 means feasible
                starting_solution=None,
                true_fitness_function=None,
                include_start_nodes: bool = True,
                target_stop=None):
    """
    Build a constrained LON (CoLON) using Deb's constraint-handling preorder.

    Returns:
      local_optima: List[Tuple[int,...]]
      fitness_values: List[float]  (objective at each optimum)
      edges_list: List[(src_tuple, dst_tuple, weight)]
      optima_feasibility: List[int]   (1 if optimum feasible, else 0) aligned with local_optima
      edge_feasibility: List[int]     (1 if edge target is feasible, else 0) aligned with edges_list
      neighbour_feasibility: List[float] (proportion 0..1 of feasible neighbours at n_flips_mut), aligned with local_optima
    """
    if violation_function is None:
        raise ValueError("violation_function is required (returns scalar v(x); <=0 means feasible).")

    # 1) Create Fitness and Individual classes if not existing
    # Check if CustomFitness exists with matching weights; recreate if weights differ
    if hasattr(creator, "CustomFitness"):
        if creator.CustomFitness.weights != weights:
            del creator.CustomFitness
            del creator.Individual
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
        return fa > fb

    # ---- Evaluators (objective & violation) ----
    def evaluate_objective(ind):
        val = fitness_function[0](ind, **fitness_function[1])
        return val if isinstance(val, tuple) else (val,)

    def evaluate_violation(ind):
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
    toolbox.register("mutate", (lambda ind: mutate_function[0](ind, **mutate_function[1]))
                     if mutate_function else (lambda ind: ind))
    toolbox.register("perturb", (lambda ind: perturb_function[0](ind, **perturb_function[1]))
                     if perturb_function else (lambda ind: ind))

    # 4) Create and evaluate a starting solution
    if starting_solution is not None:
        individual = creator.Individual(starting_solution)
    else:
        individual = creator.Individual([toolbox.attribute() for _ in range(len_sol)])
    evaluate_both(individual)

    # 5) Data
    local_optima = []
    fitness_values = []
    edges = {}
    prev_local_opt = None
    prev_fitness = None

    # caches for feasibility annotations (avoid recomputation, keep consistency)
    opt_index = {}                 # opt tuple -> index in local_optima
    opt_feas_map = {}              # opt tuple -> 1/0 (feasible flag)
    neigh_feas_prop_map = {}       # opt tuple -> float in [0,1]

    # helper: compute & cache neighbour feasibility proportion (0..1)
    def compute_neighbour_feasibility_prop(opt_tuple):
        if opt_tuple in neigh_feas_prop_map:
            return neigh_feas_prop_map[opt_tuple]
        opt_ind = creator.Individual(list(opt_tuple))
        neighbours = generate_bit_flip_combinations(opt_ind, n_flips_mut)
        if len(neighbours) == 0:
            prop = 0.0
        else:
            feas = 0
            for nb in neighbours:
                nb_violation = toolbox.violate(nb)
                if nb_violation <= 0.0:
                    feas += 1
            prop = feas / float(len(neighbours))
        neigh_feas_prop_map[opt_tuple] = prop
        return prop

    # 6) Basin-hopping
    pert_attempt = 0

    while pert_attempt < pert_attempts:
        if improv_method == 'best':
            # Best-improvement local search under Deb’s preorder
            improvement = True
            while improvement:
                # code block to record starting solutions
                if include_start_nodes:
                    start_tuple = tuple(individual)
                    if start_tuple not in opt_index:
                        local_optima.append(start_tuple)
                        fitness_values.append(individual.fitness.values[0])
                        opt_index[start_tuple] = len(local_optima) - 1
                        opt_feas_map[start_tuple] = 1 if is_feasible(individual) else 0
                        _ = compute_neighbour_feasibility_prop(start_tuple)
                # end of code block to record starting solutions

                mutants = generate_bit_flip_combinations(individual, n_flips_mut)

                # Evaluate neighbours (both f and v)
                for m in mutants:
                    if hasattr(m.fitness, "values"):
                        del m.fitness.values
                    m.fitness.values = toolbox.evaluate(m)
                    m.violation = toolbox.violate(m)

                # Pick the best among neighbours + current
                candidates = mutants + [individual]
                best_mutant = candidates[0]
                for cand in candidates[1:]:
                    if deb_better(cand, best_mutant):
                        best_mutant = cand

                # Move only if strictly better under Deb
                if deb_better(best_mutant, individual):
                    individual[:] = best_mutant
                    if hasattr(individual.fitness, "values"):
                        del individual.fitness.values
                    evaluate_both(individual)
                else:
                    improvement = False

            # Reached a local (Deb) optimum
            current_local_opt = tuple(individual)
            current_fitness = individual.fitness.values[0]

            # Add if new; also cache feasibility & neighbour feasibility
            if current_local_opt not in opt_index:
                idx = len(local_optima)
                opt_index[current_local_opt] = idx
                local_optima.append(current_local_opt)
                fitness_values.append(current_fitness)
                opt_feas_map[current_local_opt] = 1 if is_feasible(individual) else 0
                _ = compute_neighbour_feasibility_prop(current_local_opt)

            # Edge from previous optimum to current (if changed)
            if prev_local_opt is not None and prev_local_opt != current_local_opt:
                edges[(prev_local_opt, current_local_opt)] = edges.get((prev_local_opt, current_local_opt), 0) + 1

            prev_local_opt = current_local_opt
            prev_fitness = current_fitness

            if target_stop is not None and current_fitness >= target_stop and (opt_feas_map[current_local_opt] == 1):
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

    # 7) Pack edges and edge feasibility (target feasibility)
    edges_list = []
    edge_feasibility = []
    for (source, target), weight in edges.items():
        edges_list.append((source, target, weight))
        feas = opt_feas_map.get(target)
        if feas is None:
            # Shouldn't happen; compute once if needed
            tmp_ind = creator.Individual(list(target))
            tmp_violation = evaluate_violation(tmp_ind)
            feas = 1 if tmp_violation <= 0.0 else 0
            opt_feas_map[target] = feas
        edge_feasibility.append(int(feas))

    # Align feasibility lists with local_optima order
    optima_feasibility = [opt_feas_map[opt] for opt in local_optima]
    neighbour_feasibility = [neigh_feas_prop_map[opt] for opt in local_optima]

    return (local_optima, fitness_values, edges_list,
            optima_feasibility, edge_feasibility, neighbour_feasibility)
