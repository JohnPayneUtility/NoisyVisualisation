"""NSGA-II."""

# IMPORTS
import random
import pydoc

from deap import tools

from .base import OptimisationAlgorithm

# ==============================
# Evolutionary Algorithm Subclasses
# ==============================

class NSGA2(OptimisationAlgorithm):
    def __init__(
        self,
        pop_size: int = 100,
        cxpb: float = 0.9,
        mutpb: float = 0.1,
        mate_op=None,            # can be callable OR "deap.tools.cxTwoPoint"
        mutate_op=None,          # can be callable OR "deap.tools.mutFlipBit"
        mutate_params=None,      # keep this name to match your resolver
        mutate_kwargs=None,      # optional alias
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = "NSGA-II"
        self.type = "NSGA-II"

        self.pop_size = int(pop_size)
        self.cxpb = float(cxpb)
        self.mutpb = float(mutpb)

        self.gens = 0
        self.evals = 0

        # --- resolve operator references if they come in as strings/DictConfig ---
        def _resolve_callable(x):
            if x is None:
                return None
            # Hydra may pass OmegaConf nodes; str() gives dotted path nicely
            if not callable(x):
                x = str(x)
                obj = pydoc.locate(x)
                if obj is None or not callable(obj):
                    raise TypeError(f"Operator '{x}' could not be resolved to a callable.")
                return obj
            return x

        mate_fn = _resolve_callable(mate_op)
        mut_fn  = _resolve_callable(mutate_op)

        self.toolbox.register("select", tools.selNSGA2)

        if mate_fn is not None:
            self.toolbox.register("mate", mate_fn)

        # accept either mutate_params (your resolver) or mutate_kwargs
        if mutate_kwargs is None and mutate_params is not None:
            mutate_kwargs = dict(mutate_params)
        mutate_kwargs = mutate_kwargs or {}

        if mut_fn is not None:
            self.toolbox.register("mutate", mut_fn, **mutate_kwargs)

        # init + crowding distance
        self.initialise_population(pop_size=self.pop_size)
        self.population = self.toolbox.select(self.population, len(self.population))

        self.record_state_pareto(self.population)

    def perform_generation(self):
        offspring = tools.selTournamentDCD(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for ind in offspring:
            if random.random() < self.mutpb:
                out = self.toolbox.mutate(ind)
                if isinstance(out, tuple):
                    ind = out[0]
                del ind.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = self.toolbox.evaluate(ind)
        self.evals += len(invalid)

        self.population = self.toolbox.select(self.population + offspring, self.pop_size)
