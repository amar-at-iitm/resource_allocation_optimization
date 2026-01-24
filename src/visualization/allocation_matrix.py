"""Resource-Task Allocation Matrix Heatmap Visualization."""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.schemas import AllocationSolution, ProblemInstance


def plot_allocation_matrix(
    solution: AllocationSolution,
    instance: ProblemInstance,
    save_path: Optional[Path | str] = None,
    title: str = "Resource-Period Workload Allocation Matrix"
) -> plt.Figure:
    """Generates a heatmap displaying workload assigned to each resource across time periods."""
    resources = sorted(list(instance.resources.keys()))
    periods = list(range(instance.periods))

    # Construct matrix: (Resources x Periods)
    matrix = np.zeros((len(resources), len(periods)))
    r_to_idx = {r_id: idx for idx, r_id in enumerate(resources)}

    for assign in solution.assignments:
        if assign.resource_id in r_to_idx and assign.period < len(periods):
            r_idx = r_to_idx[assign.resource_id]
            matrix[r_idx, assign.period] += assign.assigned_work

    fig, ax = plt.subplots(figsize=(max(8, len(periods) * 0.8), max(6, len(resources) * 0.45)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Assigned Workload (Hours)"},
        xticklabels=[f"P_{t}" for t in periods],
        yticklabels=resources,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray"
    )

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xlabel("Planning Horizon Period", fontsize=11)
    ax.set_ylabel("Resource ID", fontsize=11)
    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved allocation matrix heatmap to: {out_file}")

    return fig

