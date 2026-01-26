"""Visual analytics for scenario comparison, demand vs capacity, and trade-off curves."""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.schemas import AllocationSolution, ProblemInstance


def plot_demand_vs_capacity(
    instance: ProblemInstance,
    solution: Optional[AllocationSolution] = None,
    save_path: Optional[Path | str] = None,
    title: str = "Aggregate Demand vs. Available Capacity Across Planning Horizon"
) -> plt.Figure:
    """Plots period-by-period Demand vs Available Capacity area/line chart highlighting buffers and deficit."""
    periods = list(range(instance.periods))
    
    # Available total nominal capacity per period
    total_capacity = [
        sum(r.capacity_per_period for r in instance.resources.values())
        for _ in periods
    ]
    
    # Overtime capacity limit
    max_capacity_with_ot = [
        sum(r.capacity_per_period + r.max_overtime_per_period for r in instance.resources.values())
        for _ in periods
    ]
    
    # Actual scheduled work from solution if provided, else aggregate potential task demand
    scheduled_work = [0.0] * instance.periods
    if solution and solution.assignments:
        for assign in solution.assignments:
            if 0 <= assign.period < instance.periods:
                scheduled_work[assign.period] += assign.assigned_work
    else:
        # Fallback: estimate demand distributed evenly over duration
        for task in instance.tasks.values():
            work_per_p = task.resource_demand / max(1, task.duration)
            for p in range(min(task.duration, instance.periods)):
                scheduled_work[p] += work_per_p

    fig, ax = plt.subplots(figsize=(10, 5.5))

    p_labels = [f"P_{t}" for t in periods]
    ax.plot(p_labels, total_capacity, label="Nominal Regular Capacity", color="#2b5c8f", linewidth=2.5, linestyle="--")
    ax.plot(p_labels, max_capacity_with_ot, label="Max Capacity (with Overtime)", color="#d95f02", linewidth=2.0, linestyle=":")
    ax.fill_between(p_labels, total_capacity, max_capacity_with_ot, color="#fdae61", alpha=0.25, label="Overtime Buffer Zone")
    
    ax.bar(p_labels, scheduled_work, color="#41b6c4", alpha=0.75, width=0.45, label="Scheduled / Assigned Demand")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Planning Horizon Period", fontsize=11)
    ax.set_ylabel("Workload (Hours)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved demand vs capacity plot to: {out_file}")

    return fig


def plot_scenario_comparison(
    csv_path: Optional[Path | str] = "results/scenario_results/uncertainty_matrix_results.csv",
    df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path | str] = None,
    title: str = "Performance & Cost Comparison Across Uncertainty Scenarios"
) -> plt.Figure:
    """Plots grouped bar charts comparing Total Cost and Demand Satisfaction across scenarios."""
    if df is None:
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Scenario results CSV not found at {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Scenario names formatted nicely
    display_names = [s.replace("Scenario_", "").replace("_", " ") for s in df["scenario"]]
    df = df.copy()
    df["display_name"] = display_names

    palette = sns.color_palette("Blues_r", n_colors=len(df))

    # Plot 1: Total Cost
    bars1 = ax1.bar(df["display_name"], df["total_cost"], color=palette, edgecolor="black", linewidth=0.8)
    ax1.set_title("Total Operational Cost by Scenario", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Cost ($)", fontsize=11)
    ax1.set_xticks(range(len(df["display_name"])))
    ax1.set_xticklabels(df["display_name"], rotation=30, ha="right")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + (yval * 0.015), f"${yval:,.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Plot 2: Demand Satisfaction %
    bars2 = ax2.bar(df["display_name"], df["demand_satisfaction_percent"], color="#74c476", edgecolor="black", linewidth=0.8)
    ax2.set_title("Demand Satisfaction Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Satisfaction Rate (%)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(range(len(df["display_name"])))
    ax2.set_xticklabels(df["display_name"], rotation=30, ha="right")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved scenario comparison plot to: {out_file}")

    return fig


def plot_tradeoff_curve(
    csv_path: Optional[Path | str] = "results/scenario_results/uncertainty_matrix_results.csv",
    df: Optional[pd.DataFrame] = None,
    save_path: Optional[Path | str] = "results/figures/scenario_tradeoffs.png",
    title: str = "Cost vs. Demand Satisfaction Trade-Off Analysis"
) -> plt.Figure:
    """Plots Pareto trade-off curve between Operational Cost and Demand Satisfaction Rate."""
    if df is None:
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Results CSV not found at {csv_path}")

    fig, ax = plt.subplots(figsize=(8.5, 6))

    scatter = ax.scatter(
        df["demand_satisfaction_percent"],
        df["total_cost"],
        c=df["solve_time_seconds"] if "solve_time_seconds" in df else None,
        cmap="viridis",
        s=160,
        edgecolors="black",
        linewidths=1.2,
        zorder=3
    )
    
    if "solve_time_seconds" in df:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Solve Time (seconds)", fontsize=10)

    for _, row in df.iterrows():
        label = row["scenario"].replace("Scenario_", "").replace("_", " ")
        ax.annotate(
            label,
            (row["demand_satisfaction_percent"], row["total_cost"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold"
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Demand Satisfaction Rate (%)", fontsize=11)
    ax.set_ylabel("Total Operational Cost ($)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved trade-off curve to: {out_file}")

    return fig
