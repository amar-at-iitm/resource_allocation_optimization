"""Genetic Algorithm for Multi-Period Resource Allocation and Scheduling."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance
from src.data.config_loader import ModelConfig, load_model_config
from src.heuristics.base import BaseMetaheuristic
from src.heuristics.encoding import ScheduleChromosome, ScheduleDecoder, generate_random_chromosome


class GeneticAlgorithmScheduler(BaseMetaheuristic):
    """Genetic Algorithm using topological order crossover and skilled resource mutation."""

    def __init__(
        self,
        instance: ProblemInstance,
        config: Optional[ModelConfig] = None,
        seed: int = 42
    ):
        super().__init__(instance, config, seed)
        self.decoder = ScheduleDecoder(instance, self.config)
        self.meta_cfg = self.config.metaheuristics
        self.skill_coverage = instance.get_skill_coverage()

    def _crossover_order(
        self,
        p1: ScheduleChromosome,
        p2: ScheduleChromosome
    ) -> Tuple[ScheduleChromosome, ScheduleChromosome]:
        """Order Crossover (OX) for task ordering and uniform crossover for resource map."""
        n_tasks = len(p1.task_order)
        if n_tasks < 2 or self.rng.random() > self.meta_cfg.crossover_rate:
            return p1.copy(), p2.copy()

        # Two crossover cut points
        i1, i2 = sorted(self.rng.choice(range(n_tasks), size=2, replace=False))

        # Offspring 1
        c1_tasks = [None] * n_tasks
        c1_tasks[i1:i2+1] = p1.task_order[i1:i2+1]
        p2_fill = [t for t in p2.task_order if t not in c1_tasks[i1:i2+1]]
        fill_idx = 0
        for idx in range(n_tasks):
            if c1_tasks[idx] is None:
                c1_tasks[idx] = p2_fill[fill_idx]
                fill_idx += 1

        # Offspring 2
        c2_tasks = [None] * n_tasks
        c2_tasks[i1:i2+1] = p2.task_order[i1:i2+1]
        p1_fill = [t for t in p1.task_order if t not in c2_tasks[i1:i2+1]]
        fill_idx = 0
        for idx in range(n_tasks):
            if c2_tasks[idx] is None:
                c2_tasks[idx] = p1_fill[fill_idx]
                fill_idx += 1

        # Uniform crossover for resource assignment
        c1_res: Dict[str, str] = {}
        c2_res: Dict[str, str] = {}
        for t_id in p1.resource_map:
            if self.rng.random() < 0.5:
                c1_res[t_id] = p1.resource_map[t_id]
                c2_res[t_id] = p2.resource_map.get(t_id, p1.resource_map[t_id])
            else:
                c1_res[t_id] = p2.resource_map.get(t_id, p1.resource_map[t_id])
                c2_res[t_id] = p1.resource_map[t_id]

        child1 = ScheduleChromosome(c1_tasks, c1_res)
        child2 = ScheduleChromosome(c2_tasks, c2_res)
        child1.repair_precedence(self.instance)
        child2.repair_precedence(self.instance)
        return child1, child2

    def _mutate(self, chromosome: ScheduleChromosome) -> None:
        """Applies swap mutation on task order and alternative skilled resource mutation."""
        if self.rng.random() < self.meta_cfg.mutation_rate:
            # 1. Swap mutation on task ordering
            if len(chromosome.task_order) >= 2:
                idx1, idx2 = self.rng.choice(range(len(chromosome.task_order)), size=2, replace=False)
                chromosome.task_order[idx1], chromosome.task_order[idx2] = (
                    chromosome.task_order[idx2],
                    chromosome.task_order[idx1]
                )
                chromosome.repair_precedence(self.instance)

        # 2. Resource mutation
        for t_id, task in self.instance.tasks.items():
            if self.rng.random() < (self.meta_cfg.mutation_rate / 2.0):
                qualified = self.skill_coverage.get(task.required_skill, [])
                if qualified and len(qualified) > 1:
                    chromosome.resource_map[t_id] = str(self.rng.choice(qualified))

    def _tournament_selection(
        self,
        population: List[ScheduleChromosome],
        costs: List[float]
    ) -> ScheduleChromosome:
        """Tournament selection picking the individual with the lowest cost."""
        k = self.meta_cfg.tournament_size
        indices = self.rng.choice(range(len(population)), size=min(k, len(population)), replace=False)
        best_idx = min(indices, key=lambda i: costs[i])
        return population[best_idx]

    def solve(
        self,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        verbose: bool = False
    ) -> AllocationSolution:
        """Executes Genetic Algorithm evolution."""
        start_time = time.time()
        pop_size = population_size or self.meta_cfg.population_size
        n_gens = generations or self.meta_cfg.generations
        elite_count = max(1, int(pop_size * 0.05))

        # 1. Initialize population
        population = [
            generate_random_chromosome(self.instance, self.rng)
            for _ in range(pop_size)
        ]
        costs = [self.decoder.decode(chrom).objective_value for chrom in population]

        best_idx = int(np.argmin(costs))
        self.best_cost = costs[best_idx]
        best_chrom = population[best_idx].copy()
        self.best_solution = self.decoder.decode(best_chrom)

        self.convergence_history = []

        # 2. Evolution Loop
        for gen in range(n_gens):
            # Sort population by cost (ascending, since minimizing)
            sorted_indices = np.argsort(costs)
            population = [population[i] for i in sorted_indices]
            costs = [costs[i] for i in sorted_indices]

            # Elitism: retain top individuals
            new_population: List[ScheduleChromosome] = [population[i].copy() for i in range(elite_count)]

            # Generate rest of new population
            while len(new_population) < pop_size:
                p1 = self._tournament_selection(population, costs)
                p2 = self._tournament_selection(population, costs)
                c1, c2 = self._crossover_order(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < pop_size:
                    new_population.append(c2)

            population = new_population
            costs = [self.decoder.decode(chrom).objective_value for chrom in population]

            current_best_idx = int(np.argmin(costs))
            current_best_cost = costs[current_best_idx]

            if current_best_cost < self.best_cost:
                self.best_cost = current_best_cost
                best_chrom = population[current_best_idx].copy()
                self.best_solution = self.decoder.decode(best_chrom)

            avg_cost = float(np.mean(costs))
            worst_cost = float(np.max(costs))

            self.convergence_history.append({
                "generation": gen + 1,
                "best_cost": round(self.best_cost, 2),
                "avg_cost": round(avg_cost, 2),
                "worst_cost": round(worst_cost, 2)
            })

            if verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen+1}/{n_gens} | Best Cost: {self.best_cost:.2f} | Avg: {avg_cost:.2f}")

        elapsed = time.time() - start_time
        assert self.best_solution is not None
        self.best_solution.solve_time_seconds = round(elapsed, 4)
        self.best_solution.status = "HEURISTIC_FEASIBLE"
        return self.best_solution

