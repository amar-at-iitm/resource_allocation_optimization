"""Runner script for Scalability Analysis and Exact vs. Metaheuristic Comparison."""

import argparse
import sys
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config_loader import load_experiment_config, load_model_config
from src.data.generator import generate_instance
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.heuristics.genetic_algorithm import GeneticAlgorithmScheduler
from src.heuristics.simulated_annealing import SimulatedAnnealingScheduler


def run_heuristic_comparison(
    scale: str = "large",
    iterations: int = 50,
    exact_timelimit: float = 60.0,
    num_replications: int = 2,
    output_dir: Path | str = "results/heuristic_results"
) -> pd.DataFrame:
    """Compares exact HiGHS optimization against GA and SA across problem instances."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exp_config = load_experiment_config()
    model_config = load_model_config()

    sc_spec = exp_config.scales.get(scale)
    if not sc_spec:
        raise ValueError(f"Scale '{scale}' not defined in experiment_config.yaml")

    print(f"\n==================================================================")
    print(f"Running Exact vs. Metaheuristic Comparison: {scale.upper()} Scale")
    print(f"Specs: {sc_spec.n_resources} Resources, {sc_spec.n_tasks} Tasks, {sc_spec.n_periods} Periods")
    print(f"Iterations/Gens: {iterations} | Exact Time Limit: {exact_timelimit}s | Reps: {num_replications}")
    print(f"==================================================================")

    comparison_records: List[Dict[str, Any]] = []

    for repl in range(num_replications):
        seed = exp_config.experiment.random_seed + repl
        print(f"\n--- Replication {repl + 1}/{num_replications} (Seed: {seed}) ---")

        inst = generate_instance(
            n_resources=sc_spec.n_resources,
            n_tasks=sc_spec.n_tasks,
            n_periods=sc_spec.n_periods,
            seed=seed,
            instance_id=f"{scale}_comp_seed{seed}"
        )

        # -------------------------------------------------------------
        # 1. Exact Optimization (HiGHS)
        # -------------------------------------------------------------
        print("1. Running Exact Solver (HiGHS)...")
        exact_start = time.time()
        exact_model = MultiPeriodSchedulingModel(inst, config=model_config)
        exact_sol = exact_model.solve(time_limit=exact_timelimit, log_to_console=False)
        exact_runtime = time.time() - exact_start
        exact_cost = exact_sol.objective_value
        print(f"   HiGHS: Status={exact_sol.status} | Cost={exact_cost} | Runtime={exact_runtime:.2f}s")

        # -------------------------------------------------------------
        # 2. Genetic Algorithm (GA)
        # -------------------------------------------------------------
        print("2. Running Genetic Algorithm (GA)...")
        ga = GeneticAlgorithmScheduler(inst, config=model_config, seed=seed)
        ga_sol = ga.solve(population_size=40, generations=iterations, verbose=False)
        ga_cost = ga_sol.objective_value
        ga_runtime = ga_sol.solve_time_seconds
        ga_gap = (((ga_cost - exact_cost) / exact_cost) * 100.0) if exact_cost and exact_cost > 0 else 0.0
        print(f"   GA: Cost={ga_cost} | Runtime={ga_runtime:.2f}s | Gap={ga_gap:.2f}%")

        # -------------------------------------------------------------
        # 3. Simulated Annealing (SA)
        # -------------------------------------------------------------
        print("3. Running Simulated Annealing (SA)...")
        sa = SimulatedAnnealingScheduler(inst, config=model_config, seed=seed)
        sa_sol = sa.solve(
            initial_temp=500.0,
            cooling_rate=0.92,
            min_temp=1.0,
            max_iterations_per_temp=max(1, iterations // 5),
            verbose=False
        )
        sa_cost = sa_sol.objective_value
        sa_runtime = sa_sol.solve_time_seconds
        sa_gap = (((sa_cost - exact_cost) / exact_cost) * 100.0) if exact_cost and exact_cost > 0 else 0.0
        print(f"   SA: Cost={sa_cost} | Runtime={sa_runtime:.2f}s | Gap={sa_gap:.2f}%")

        # Log records
        comparison_records.extend([
            {
                "instance_id": inst.instance_id,
                "scale": scale,
                "seed": seed,
                "algorithm": "HiGHS_Exact",
                "status": exact_sol.status,
                "objective_cost": exact_cost,
                "runtime_seconds": exact_runtime,
                "optimality_gap_percent": 0.0 if exact_sol.status == "OPTIMAL" else None,
                "scheduled_tasks": exact_sol.summary_kpis.get("total_assigned_tasks", 0),
                "unmet_tasks": exact_sol.summary_kpis.get("total_unmet_tasks", 0),
                "utilization_percent": exact_sol.summary_kpis.get("capacity_utilization_percent", 0.0)
            },
            {
                "instance_id": inst.instance_id,
                "scale": scale,
                "seed": seed,
                "algorithm": "GeneticAlgorithm",
                "status": ga_sol.status,
                "objective_cost": ga_cost,
                "runtime_seconds": ga_runtime,
                "optimality_gap_percent": round(ga_gap, 2),
                "scheduled_tasks": ga_sol.summary_kpis.get("total_assigned_tasks", 0),
                "unmet_tasks": ga_sol.summary_kpis.get("total_unmet_tasks", 0),
                "utilization_percent": ga_sol.summary_kpis.get("capacity_utilization_percent", 0.0)
            },
            {
                "instance_id": inst.instance_id,
                "scale": scale,
                "seed": seed,
                "algorithm": "SimulatedAnnealing",
                "status": sa_sol.status,
                "objective_cost": sa_cost,
                "runtime_seconds": sa_runtime,
                "optimality_gap_percent": round(sa_gap, 2),
                "scheduled_tasks": sa_sol.summary_kpis.get("total_assigned_tasks", 0),
                "unmet_tasks": sa_sol.summary_kpis.get("total_unmet_tasks", 0),
                "utilization_percent": sa_sol.summary_kpis.get("capacity_utilization_percent", 0.0)
            }
        ])

    df = pd.DataFrame.from_records(comparison_records)
    csv_file = out_path / "large_scale_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved comparison results to: {csv_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exact vs. Metaheuristic Scalability Experiments")
    parser.add_argument("--scale", type=str, choices=["small", "medium", "large"], default="large",
                        help="Scale of benchmark instances")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations or generations for metaheuristics")
    parser.add_argument("--timelimit", type=float, default=60.0,
                        help="Exact solver timeout in seconds")
    parser.add_argument("--replications", type=int, default=2,
                        help="Number of seed replications")
    parser.add_argument("--output_dir", type=str, default="results/heuristic_results",
                        help="Output directory for CSV results")
    args = parser.parse_args()

    run_heuristic_comparison(
        scale=args.scale,
        iterations=args.iterations,
        exact_timelimit=args.timelimit,
        num_replications=args.replications,
        output_dir=args.output_dir
    )

