"""Interactive Plotly Gantt Chart for Multi-Period Resource Scheduling."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data.schemas import AllocationSolution, ProblemInstance


def plot_gantt(
    solution: AllocationSolution,
    instance: ProblemInstance,
    save_html_path: Optional[Path | str] = None,
    title: str = "Multi-Period Resource Scheduling Gantt Chart"
) -> go.Figure:
    """Generates an interactive Plotly Gantt chart showing scheduled tasks on each resource."""
    records: List[Dict[str, Any]] = []

    # Map assignments to contiguous intervals per task and resource
    # Or use task_start_times and task_completion_times with assigned resource
    task_res: Dict[str, str] = {}
    for assign in solution.assignments:
        task_res[assign.task_id] = assign.resource_id

    for t_id, start_t in solution.task_start_times.items():
        task = instance.tasks.get(t_id)
        if not task:
            continue
        comp_t = solution.task_completion_times.get(t_id, start_t + task.duration - 1)
        res_id = task_res.get(t_id, "Unassigned")

        records.append({
            "Task": t_id,
            "Task_Name": task.name,
            "Resource": res_id,
            "Start": start_t,
            "Finish": comp_t + 1,  # End interval for plotting
            "Duration": task.duration,
            "Demand": task.resource_demand,
            "Skill": task.required_skill,
            "Predecessors": ", ".join(task.predecessors) if task.predecessors else "None"
        })

    if not records:
        # Empty figure if no tasks scheduled
        fig = go.Figure()
        fig.update_layout(title="No Scheduled Tasks in Solution")
        return fig

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by=["Resource", "Start"]).reset_index(drop=True)

    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Resource",
        color="Skill",
        text="Task",
        hover_data=["Task_Name", "Duration", "Demand", "Predecessors"],
        title=title,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_yaxes(autorange="reversed")  # First resource on top
    fig.update_xaxes(
        type="linear",
        title="Planning Period",
        dtick=1,
        range=[0, max(instance.periods, df["Finish"].max() + 1)]
    )
    fig.update_traces(textposition="inside", marker_line_color="rgb(50,50,50)", marker_line_width=1.5, opacity=0.9)
    fig.update_layout(
        xaxis_title="Planning Horizon Period",
        yaxis_title="Resource",
        legend_title="Required Skill",
        bargap=0.2,
        height=max(450, len(instance.resources) * 45)
    )

    if save_html_path:
        out_file = Path(save_html_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_file))
        print(f"Saved interactive Gantt chart to: {out_file}")

    return fig

