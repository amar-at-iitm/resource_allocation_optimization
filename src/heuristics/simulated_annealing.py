"""Simulated Annealing algorithm for Multi-Period Resource Allocation and Scheduling."""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance
from src.data.config_loader import ModelConfig, load_model_config
from src.heuristics.base import BaseMetaheuristic
from src.heuristics.encoding import ScheduleChromosome, ScheduleDecoder, generate_random_chromosome


class SimulatedAnnealingScheduler(BaseMetaheuristic):
    """Simulated Annealing with adaptive neighborhood moves and geometric cooling."""

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

    def _generate_neighbor(self, current: ScheduleChromosome) -> ScheduleChromosome:
        """Generates a neighboring solution via swap, shift, or resource reassignment."""
        neighbor = current.copy()
        move_type = self.rng.choice(["swap", "shift", "reassign"])

        n_tasks = len(neighbor.task_order)
        if n_tasks < 2:
            move_type = "reassign"

        if move_type == "swap" and n_tasks >= 2:
            i, j = self.rng.choice(range(n_tasks), size=2, replace=False)
            neighbor.task_order[i], neighbor.task_order[j] = neighbor.task_order[j], neighbor.task_order[i]
            neighbor.repair_precedence(self.instance)

        elif move_type == "shift" and n_tasks >= 2:
            i, j = self.rng.choice(range(n_tasks), size=2, replace=False)
            task = neighbor.task_order.pop(i)
            neighbor.task_order.insert(j, task)
            neighbor.repair_precedence(self.instance)

        elif move_type == "reassign":
            # Pick a task and assign to another qualified resource
            t_id = str(self.rng.choice(list(self.instance.tasks.keys())))
            task = self.instance.tasks[t_id]
            qualified = self.skill_coverage.get(task.required_skill, [])
            if qualified:
                neighbor.resource_map[t_id] = str(self.rng.choice(qualified))

        return neighbor

    def solve(
        self,
        initial_temp: Optional[float] = None,
        cooling_rate: Optional[float] = None,
        min_temp: Optional[float] = None,
        max_iterations_per_temp: int = 10,
        verbose: bool = False
    ) -> AllocationSolution:
        """Runs the Simulated Annealing trajectory search."""
        start_time = time.time()
        temp = initial_temp or self.meta_cfg.sa_initial_temp
        alpha = cooling_rate or self.meta_cfg.sa_cooling_rate
        t_min = min_temp or self.meta_cfg.sa_min_temp

        # 1. Initial State
        current = generate_random_chromosome(self.instance, self.rng)
        current_cost = self.decoder.decode(current).objective_value

        self.best_cost = current_cost
        best_chrom = current.copy()
        self.best_solution = self.decoder.decode(best_chrom)

        self.convergence_history = []
        iteration = 0

        # 2. Annealing Loop
        while temp > t_min:
            for _ in range(max_iterations_per_temp):
                iteration += 1
                neighbor = self._generate_neighbor(current)
                neighbor_cost = self.decoder.decode(neighbor).objective_value

                delta = neighbor_cost - current_cost

                # Metropolis acceptance criterion (minimizing cost)
                if delta < 0 or self.rng.random() < math.exp(-delta / max(1e-6, temp)):
                    current = neighbor
                    current_cost = neighbor_cost

                    if current_cost < self.best_cost:
                        self.best_cost = current_cost
                        best_chrom = current.copy()
                        self.best_solution = self.decoder.decode(best_chrom)

            self.convergence_history.append({
                "iteration": iteration,
                "temperature": round(temp, 4),
                "current_cost": round(current_cost, 2),
                "best_cost": round(self.best_cost, 2)
            })

            temp *= alpha

            if verbose and iteration % 100 == 0:
                print(f"Iter {iteration} | Temp: {temp:.2f} | Best Cost: {self.best_cost:.2f}")

        elapsed = time.time() - start_time
        assert self.best_solution is not None
        self.best_solution.solve_time_seconds = round(elapsed, 4)
        self.best_solution.status = "SA_FEASIBLE"
        return self.best_solution

