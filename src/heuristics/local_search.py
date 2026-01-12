"""Greedy local search hill-climbing scheduler."""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance
from src.data.config_loader import ModelConfig, load_model_config
from src.heuristics.base import BaseMetaheuristic
from src.heuristics.encoding import ScheduleChromosome, ScheduleDecoder, generate_random_chromosome


class LocalSearchScheduler(BaseMetaheuristic):
    """Iterated Local Search using best-improvement neighborhood descent."""

    def __init__(
        self,
        instance: ProblemInstance,
        config: Optional[ModelConfig] = None,
        seed: int = 42
    ):
        super().__init__(instance, config, seed)
        self.decoder = ScheduleDecoder(instance, self.config)
        self.skill_coverage = instance.get_skill_coverage()

    def solve(
        self,
        max_steps: int = 100,
        verbose: bool = False
    ) -> AllocationSolution:
        """Runs greedy steepest-descent local search."""
        start_time = time.time()

        current = generate_random_chromosome(self.instance, self.rng)
        current_cost = self.decoder.decode(current).objective_value
        self.best_cost = current_cost
        best_chrom = current.copy()

        self.convergence_history = []

        for step in range(max_steps):
            improved = False
            best_neighbor_chrom = None
            best_neighbor_cost = current_cost

            # Explore all single-task resource reassignments
            for t_id, task in self.instance.tasks.items():
                qualified = self.skill_coverage.get(task.required_skill, [])
                for r_cand in qualified:
                    if r_cand == current.resource_map.get(t_id):
                        continue
                    neighbor = current.copy()
                    neighbor.resource_map[t_id] = r_cand
                    cost = self.decoder.decode(neighbor).objective_value

                    if cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbor_chrom = neighbor
                        improved = True

            # Explore pairwise task swaps in order
            n_tasks = len(current.task_order)
            for i in range(min(5, n_tasks)):
                for j in range(i + 1, min(6, n_tasks)):
                    neighbor = current.copy()
                    neighbor.task_order[i], neighbor.task_order[j] = neighbor.task_order[j], neighbor.task_order[i]
                    neighbor.repair_precedence(self.instance)
                    cost = self.decoder.decode(neighbor).objective_value

                    if cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbor_chrom = neighbor
                        improved = True

            if improved and best_neighbor_chrom is not None:
                current = best_neighbor_chrom
                current_cost = best_neighbor_cost
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    best_chrom = current.copy()
            else:
                # Local optimum reached
                break

            self.convergence_history.append({
                "step": step + 1,
                "cost": round(self.best_cost, 2)
            })

        elapsed = time.time() - start_time
        self.best_solution = self.decoder.decode(best_chrom)
        self.best_solution.solve_time_seconds = round(elapsed, 4)
        self.best_solution.status = "LOCAL_OPTIMUM"
        return self.best_solution

