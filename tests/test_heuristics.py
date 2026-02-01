"""Unit and comparative tests for metaheuristics (GA, SA, Local Search)."""

import pytest
import numpy as np

from src.data.generator import generate_instance
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.heuristics.encoding import ScheduleChromosome, ScheduleDecoder, generate_random_chromosome
from src.heuristics.genetic_algorithm import GeneticAlgorithmScheduler
from src.heuristics.simulated_annealing import SimulatedAnnealingScheduler
from src.heuristics.local_search import LocalSearchScheduler


def test_chromosome_repair():
    """Tests that arbitrary task permutations are repaired to valid topological orders."""
    inst = generate_instance(n_resources=3, n_tasks=6, n_periods=8, seed=42, precedence_density=0.4)
    tasks = inst.tasks

    # Invert order deliberately to violate precedence
    inverted_tasks = list(tasks.keys())[::-1]
    res_map = {t_id: list(inst.resources.keys())[0] for t_id in tasks}
    chrom = ScheduleChromosome(inverted_tasks, res_map)
    chrom.repair_precedence(inst)

    # Check topological property: for each task, all its predecessors must appear earlier in task_order
    pos = {t_id: idx for idx, t_id in enumerate(chrom.task_order)}
    for t_id, task in tasks.items():
        for pred in task.predecessors:
            if pred in pos:
                assert pos[pred] < pos[t_id], f"Precedence broken: {pred} at {pos[pred]}, {t_id} at {pos[t_id]}"


def test_decoder_constraint_satisfaction():
    """Tests that ScheduleDecoder produces strictly valid, feasible assignments and respects precedence."""
    inst = generate_instance(n_resources=4, n_tasks=8, n_periods=10, seed=123, precedence_density=0.3)
    rng = np.random.default_rng(123)
    chrom = generate_random_chromosome(inst, rng)
    decoder = ScheduleDecoder(inst)
    sol = decoder.decode(chrom)

    assert sol.is_feasible is True

    # 1. Skill compatibility check
    for assign in sol.assignments:
        res = inst.resources[assign.resource_id]
        task = inst.tasks[assign.task_id]
        assert task.required_skill in res.skills

    # 2. Precedence check
    for t_id, task in inst.tasks.items():
        if t_id in sol.task_start_times:
            s_t = sol.task_start_times[t_id]
            for pred in task.predecessors:
                if pred in sol.task_completion_times:
                    c_p = sol.task_completion_times[pred]
                    assert s_t > c_p, f"Precedence violation: {pred} ended at {c_p}, but {t_id} started at {s_t}"


def test_ga_determinism_and_convergence():
    """Tests that GA is reproducible with fixed seed and maintains or improves best fitness."""
    inst = generate_instance(n_resources=3, n_tasks=6, n_periods=7, seed=42)

    ga1 = GeneticAlgorithmScheduler(inst, seed=99)
    sol1 = ga1.solve(population_size=15, generations=10)

    ga2 = GeneticAlgorithmScheduler(inst, seed=99)
    sol2 = ga2.solve(population_size=15, generations=10)

    # Strict reproducibility
    assert abs(sol1.objective_value - sol2.objective_value) < 1e-4

    # Best cost must be non-increasing over generations
    best_costs = [entry["best_cost"] for entry in ga1.convergence_history]
    for i in range(len(best_costs) - 1):
        assert best_costs[i+1] <= best_costs[i] + 1e-4


def test_sa_execution():
    """Tests Simulated Annealing executes and returns feasible solution."""
    inst = generate_instance(n_resources=4, n_tasks=6, n_periods=8, seed=42)
    sa = SimulatedAnnealingScheduler(inst, seed=42)
    sol = sa.solve(initial_temp=50.0, cooling_rate=0.85, min_temp=1.0, max_iterations_per_temp=5)

    assert sol.is_feasible is True
    assert sol.objective_value > 0.0
    assert len(sa.convergence_history) > 0


def test_optimality_gap_comparison():
    """Compares exact solver vs metaheuristics on a small instance to evaluate the optimality gap."""
    inst = generate_instance(n_resources=3, n_tasks=5, n_periods=7, seed=42, precedence_density=0.2)

    # Exact HiGHS
    exact_model = MultiPeriodSchedulingModel(inst)
    exact_sol = exact_model.solve(log_to_console=False)
    assert exact_sol.status == "OPTIMAL"
    exact_cost = exact_sol.objective_value

    # GA
    ga = GeneticAlgorithmScheduler(inst, seed=42)
    ga_sol = ga.solve(population_size=25, generations=20)
    ga_cost = ga_sol.objective_value

    # Gap: ((Heuristic - Exact) / Exact) * 100
    gap = ((ga_cost - exact_cost) / exact_cost) * 100.0

    print(f"\nExact Cost: {exact_cost:.2f} | GA Cost: {ga_cost:.2f} | Gap: {gap:.2f}%")
    # GA should produce a reasonable solution (heuristic >= exact optimum - epsilon)
    assert ga_cost >= exact_cost - 1e-2

