"""Master End-to-End Pipeline Runner Script.

Executes data generation, exact models, heuristics, uncertainty suite,
and visual report generation with Rich terminal formatting.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.generator import generate_instance
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.visualization.allocation_matrix import plot_allocation_matrix
from src.visualization.gantt_chart import plot_gantt
from src.visualization.scenario_comparison import (
    plot_demand_vs_capacity,
    plot_scenario_comparison,
    plot_tradeoff_curve,
)
from scripts.run_deterministic_model import run_experiments
from scripts.run_heuristics import run_heuristic_comparison
from scripts.run_uncertainty_analysis import run_uncertainty_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete optimization experimentation pipeline.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run quick lightweight test suite for verification.")
    parser.add_argument("--skip-large", action="store_true", help="Skip large-scale heuristic benchmarks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()

    console.print(Panel.fit(
        "[bold cyan]⚡ Multi-Period Resource Allocation & Scheduling Pipeline[/bold cyan]\n"
        "[italic]End-to-End Optimization, Uncertainty Analysis & Benchmarking[/italic]",
        border_style="cyan"
    ))

    start_total_time = time.perf_counter()

    # Step 1: Deterministic Optimization Experiments
    console.rule("[bold green]Stage 1: Deterministic Optimization Benchmarks[/bold green]")
    with console.status("[bold green]Running deterministic models...", spinner="dots"):
        if args.fast_dev_run:
            df_det = run_experiments(scale="small", timelimit=15.0, num_replications=1)
        else:
            df_det = run_experiments(scale="all", timelimit=60.0, num_replications=2)
    console.print(f"✔ Deterministic experiments finished. Generated {len(df_det)} records.")

    # Step 2: Metaheuristic Benchmarks (GA & SA)
    console.rule("[bold green]Stage 2: Metaheuristic Scalability Benchmarks[/bold green]")
    if args.skip_large:
        console.print("[yellow]Skipping large-scale heuristic benchmarks per --skip-large flag.[/yellow]")
        df_heur = pd.DataFrame()
    else:
        with console.status("[bold green]Running GA, SA, and Exact comparative tests...", spinner="dots"):
            if args.fast_dev_run:
                df_heur = run_heuristic_comparison(scale="medium", iterations=15, exact_timelimit=15.0, num_replications=1)
            else:
                df_heur = run_heuristic_comparison(scale="large", iterations=50, exact_timelimit=60.0, num_replications=2)
        console.print(f"✔ Heuristic comparison finished. Generated {len(df_heur)} records.")

    # Step 3: Uncertainty & Resilience Suite
    console.rule("[bold green]Stage 3: Multi-Scenario Uncertainty & Resilience Analysis[/bold green]")
    with console.status("[bold green]Running uncertainty scenarios (A, B, C, D, Stochastic, Robust)...", spinner="dots"):
        if args.fast_dev_run:
            df_unc = run_uncertainty_matrix(n_resources=6, n_tasks=8, n_periods=7, seed=args.seed)
        else:
            df_unc = run_uncertainty_matrix(n_resources=15, n_tasks=20, n_periods=10, seed=args.seed)
    console.print(f"✔ Uncertainty matrix finished. Generated {len(df_unc)} records.")

    # Step 4: Figure Generation
    console.rule("[bold green]Stage 4: Automated Figure & Report Generation[/bold green]")
    with console.status("[bold green]Rendering visualization artifacts...", spinner="dots"):
        inst_fig = generate_instance(n_resources=8, n_tasks=12, n_periods=7, seed=args.seed)
        model_fig = MultiPeriodSchedulingModel(inst_fig)
        sol_fig = model_fig.solve(time_limit=15.0)

        Path("results/figures").mkdir(parents=True, exist_ok=True)
        plot_allocation_matrix(sol_fig, inst_fig, save_path="results/figures/allocation_matrix.png")
        plot_gantt(sol_fig, inst_fig, save_html_path="results/figures/gantt_chart.html")
        plot_demand_vs_capacity(inst_fig, sol_fig, save_path="results/figures/demand_vs_capacity.png")
        plot_scenario_comparison(save_path="results/figures/scenario_comparison.png")
        plot_tradeoff_curve(save_path="results/figures/scenario_tradeoffs.png")
    console.print("✔ All figures successfully exported to [bold]results/figures/[/bold].")

    # Step 5: Summary Table Display
    console.rule("[bold cyan]Executive Summary[/bold cyan]")
    table = Table(title="Optimization Performance Highlights", show_header=True, header_style="bold magenta")
    table.add_column("Experiment Category", style="cyan", width=26)
    table.add_column("Key Scenario / Scale", style="green", width=22)
    table.add_column("Status", style="bold", width=12)
    table.add_column("Best Objective ($)", justify="right", width=18)
    table.add_column("Demand Satisfied", justify="right", width=16)

    # Highlight top records
    if not df_det.empty:
        top_det = df_det.iloc[0]
        table.add_row("Deterministic Exact", str(top_det.get("scale", "small")), str(top_det.get("solver_status", "N/A")), f"${top_det.get('objective_cost', 0):,.2f}", f"{top_det.get('demand_satisfaction_percent', 100):.1f}%")
    if not df_heur.empty:
        best_heur = df_heur.sort_values(by="objective_cost").iloc[0]
        table.add_row("Metaheuristic (GA/SA)", str(best_heur.get("algorithm", "GA")), str(best_heur.get("status", "N/A")), f"${best_heur.get('objective_cost', 0):,.2f}", "N/A")
    if not df_unc.empty:
        norm_unc = df_unc[df_unc["scenario"].str.contains("Normal")].iloc[0] if any(df_unc["scenario"].str.contains("Normal")) else df_unc.iloc[0]
        table.add_row("Scenario Analysis", str(norm_unc.get("scenario", "Scenario_A")), str(norm_unc.get("solver_status", "N/A")), f"${norm_unc.get('total_cost', 0):,.2f}", f"{norm_unc.get('demand_satisfaction_percent', 0):.1f}%")

    console.print(table)

    total_duration = time.perf_counter() - start_total_time
    console.print(Panel.fit(
        f"[bold green]✔ Complete Pipeline Finished Successfully in {total_duration:.2f} seconds![/bold green]\n"
        f"Data artifacts written to [bold]results/[/bold].",
        border_style="green"
    ))

    # Write Markdown Summary
    summary_path = ROOT_DIR / "results" / "experiment_summary.md"
    summary_md = f"""# Computational Experiment Summary Report

**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Wall-Clock Time:** {total_duration:.2f} seconds
**Mode:** {'Fast Dev Run' if args.fast_dev_run else 'Full Production Run'}

## Key Experimental Results

- **Deterministic Optimization:** Completed with {len(df_det)} evaluation runs.
- **Metaheuristic Comparison:** Evaluated {len(df_heur)} algorithm configurations.
- **Uncertainty Analysis:** Evaluated {len(df_unc)} scenario combinations across baseline, surge, disruption, stochastic, and robust formulations.
- **Visual Artifacts:** Generated Gantt charts, heatmaps, demand buffers, and Pareto trade-off curves in `results/figures/`.
"""
    summary_path.write_text(summary_md)
    console.print(f"Summary markdown generated: [italic]{summary_path}[/italic]")


if __name__ == "__main__":
    main()
