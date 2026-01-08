"""Optimization Results Extractor and Solution Serializer."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.schemas import AllocationSolution, ProblemInstance


class SolutionExtractor:
    """Extracts, analyzes, and exports optimization model solutions into DataFrames and reports."""

    def __init__(self, solution: AllocationSolution, instance: Optional[ProblemInstance] = None):
        self.solution = solution
        self.instance = instance

    def to_assignment_dataframe(self) -> pd.DataFrame:
        """Converts task assignments into a pandas DataFrame."""
        if not self.solution.assignments:
            return pd.DataFrame(columns=["resource_id", "task_id", "period", "assigned_work"])

        records = [
            {
                "resource_id": a.resource_id,
                "task_id": a.task_id,
                "period": a.period,
                "assigned_work": a.assigned_work
            }
            for a in self.solution.assignments
        ]
        df = pd.DataFrame.from_records(records)
        return df.sort_values(by=["period", "resource_id", "task_id"]).reset_index(drop=True)

    def to_resource_kpi_dataframe(self) -> pd.DataFrame:
        """Computes period-by-period workload, overtime, idle hours, and utilization per resource."""
        records: List[Dict[str, Any]] = []
        resources = self.instance.resources if self.instance else {}

        all_resources = set(self.solution.overtime.keys()) | set(self.solution.idle.keys())
        if self.instance:
            all_resources |= set(self.instance.resources.keys())

        for r_id in sorted(all_resources):
            r_obj = resources.get(r_id)
            cap = r_obj.capacity_per_period if r_obj else 8.0
            ot_periods = self.solution.overtime.get(r_id, {})
            idle_periods = self.solution.idle.get(r_id, {})

            all_p = sorted(set(ot_periods.keys()) | set(idle_periods.keys()))
            if not all_p and self.instance:
                all_p = list(range(self.instance.periods))

            for p in all_p:
                ot = ot_periods.get(p, 0.0)
                idl = idle_periods.get(p, 0.0)
                assigned = max(0.0, cap + ot - idl)
                util = round((assigned / max(1e-4, cap + ot)) * 100.0, 2)
                records.append({
                    "resource_id": r_id,
                    "period": p,
                    "assigned_hours": round(assigned, 2),
                    "regular_capacity": round(cap, 2),
                    "overtime_hours": round(ot, 2),
                    "idle_hours": round(idl, 2),
                    "utilization_percent": util
                })

        return pd.DataFrame.from_records(records)

    def to_task_schedule_dataframe(self) -> pd.DataFrame:
        """Constructs task schedule overview (start times, completion times, status)."""
        records: List[Dict[str, Any]] = []
        tasks = self.instance.tasks if self.instance else {}

        for t_id, task in tasks.items():
            is_scheduled = t_id in self.solution.task_start_times
            start_t = self.solution.task_start_times.get(t_id, None)
            comp_t = self.solution.task_completion_times.get(t_id, None)
            status = "SCHEDULED" if is_scheduled else "UNMET"

            records.append({
                "task_id": t_id,
                "name": task.name,
                "required_skill": task.required_skill,
                "duration": task.duration,
                "resource_demand": task.resource_demand,
                "priority": task.priority,
                "start_period": start_t,
                "completion_period": comp_t,
                "status": status,
                "unmet_penalty": task.unmet_penalty if not is_scheduled else 0.0
            })

        return pd.DataFrame.from_records(records)

    def get_summary_report(self) -> Dict[str, Any]:
        """Returns structured dictionary of high-level performance metrics."""
        return {
            "instance_id": self.solution.instance_id,
            "solver_status": self.solution.status,
            "objective_cost": self.solution.objective_value,
            "solve_time_seconds": self.solution.solve_time_seconds,
            "optimality_gap": self.solution.optimality_gap,
            **self.solution.summary_kpis
        }

    def export_results(
        self,
        output_dir: Path | str,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """Exports solution JSON, assignment CSV, resource KPI CSV, and task schedule CSV."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""

        exported_files: Dict[str, Path] = {}

        # 1. Solution JSON
        json_file = out_path / f"{pfx}solution.json"
        self.solution.save_json(json_file)
        exported_files["solution_json"] = json_file

        # 2. Assignments CSV
        assign_df = self.to_assignment_dataframe()
        assign_file = out_path / f"{pfx}assignments.csv"
        assign_df.to_csv(assign_file, index=False)
        exported_files["assignments_csv"] = assign_file

        # 3. Resource KPIs CSV
        if self.instance:
            kpi_df = self.to_resource_kpi_dataframe()
            kpi_file = out_path / f"{pfx}resource_kpis.csv"
            kpi_df.to_csv(kpi_file, index=False)
            exported_files["resource_kpis_csv"] = kpi_file

            task_df = self.to_task_schedule_dataframe()
            task_file = out_path / f"{pfx}task_schedule.csv"
            task_df.to_csv(task_file, index=False)
            exported_files["task_schedule_csv"] = task_file

        return exported_files

