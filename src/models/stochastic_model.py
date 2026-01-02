"""Two-Stage Stochastic Resource Allocation Optimization Model."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance, TaskAssignment
from src.data.config_loader import ModelConfig, load_model_config
from src.optimization.solver import MILPSolver


class TwoStageStochasticAllocationModel:
    """Two-Stage Stochastic Programming model for resource allocation under scenario uncertainty.
    
    First-Stage Decision: Resource contracting / shift availability reservation z[i, t].
    Second-Stage Decision: Operational task assignment x[i, j, t, s], overtime o[i, t, s],
                           idle capacity idle[i, t, s], and unmet demand u[j, t, s] per scenario.
    """

    def __init__(
        self,
        scenarios: List[ProblemInstance],
        probabilities: Optional[List[float]] = None,
        config: Optional[ModelConfig] = None
    ):
        if not scenarios:
            raise ValueError("Stochastic model requires at least one scenario.")
        self.scenarios = scenarios
        self.num_scenarios = len(scenarios)

        if probabilities is None:
            self.probabilities = [1.0 / self.num_scenarios] * self.num_scenarios
        else:
            if len(probabilities) != self.num_scenarios:
                raise ValueError("Length of probabilities must match number of scenarios.")
            tot_p = sum(probabilities)
            self.probabilities = [p / tot_p for p in probabilities]

        self.config = config or load_model_config()
        self.solver = MILPSolver()
        self.base_instance = scenarios[0]
        self.solution: Optional[AllocationSolution] = None
        self.scenario_solutions: Dict[str, Dict[str, Any]] = {}

    def build_model(self) -> None:
        """Constructs extensive form two-stage stochastic MILP."""
        resources = self.base_instance.resources
        tasks = self.base_instance.tasks
        n_periods = self.base_instance.periods
        costs = self.config.costs

        # -------------------------------------------------------------
        # 1. FIRST-STAGE DECISION VARIABLES: Base Contract z[i, t] in {0, 1}
        # Fixed cost to activate/contract resource i in period t
        # -------------------------------------------------------------
        for r_id, res in resources.items():
            fixed_reservation_cost = res.regular_cost_per_period * 0.20
            for t in range(n_periods):
                self.solver.add_variable(
                    name=f"z_{r_id}_{t}",
                    lb=0.0,
                    ub=1.0,
                    var_type="BINARY",
                    obj_coeff=fixed_reservation_cost
                )

        # -------------------------------------------------------------
        # 2. SECOND-STAGE VARIABLES (Per Scenario s)
        # Weighted by scenario probability p_s in the objective
        # -------------------------------------------------------------
        for s_idx, scen in enumerate(self.scenarios):
            p_s = self.probabilities[s_idx]

            # x[i, j, t, s]
            for r_id, res in scen.resources.items():
                for t_id, task in scen.tasks.items():
                    is_qualified = task.required_skill in res.skills
                    for t in range(n_periods):
                        var_name = f"x_{r_id}_{t_id}_{t}_s{s_idx}"
                        if self.config.constraints.enforce_skill_compatibility and not is_qualified:
                            self.solver.add_variable(name=var_name, lb=0.0, ub=0.0, var_type="BINARY", obj_coeff=0.0)
                        else:
                            hourly_rate = res.regular_cost_per_period / max(1e-3, res.capacity_per_period)
                            reg_cost = hourly_rate * task.resource_demand * costs.regular_cost_multiplier
                            self.solver.add_variable(
                                name=var_name,
                                lb=0.0,
                                ub=1.0,
                                var_type="BINARY",
                                obj_coeff=p_s * reg_cost
                            )

            # o[i, t, s]: Overtime
            for r_id, res in scen.resources.items():
                max_ot = res.max_overtime_per_period if self.config.constraints.allow_overtime else 0.0
                ot_cost = (res.overtime_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.overtime_cost_multiplier
                for t in range(n_periods):
                    self.solver.add_variable(
                        name=f"o_{r_id}_{t}_s{s_idx}",
                        lb=0.0,
                        ub=max_ot,
                        var_type="CONTINUOUS",
                        obj_coeff=p_s * ot_cost
                    )

            # idle[i, t, s]: Idle capacity
            for r_id, res in scen.resources.items():
                idle_cost = (res.idle_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.idle_cost_multiplier
                for t in range(n_periods):
                    self.solver.add_variable(
                        name=f"idle_{r_id}_{t}_s{s_idx}",
                        lb=0.0,
                        ub=res.capacity_per_period,
                        var_type="CONTINUOUS",
                        obj_coeff=p_s * idle_cost
                    )

            # y[j, s]: Completed indicator (1 if completed, 0 if unmet)
            for t_id, task in scen.tasks.items():
                self.solver.add_variable(
                    name=f"y_{t_id}_s{s_idx}",
                    lb=0.0,
                    ub=1.0,
                    var_type="BINARY",
                    obj_coeff=-p_s * task.unmet_penalty
                )

        # -------------------------------------------------------------
        # 3. SECOND-STAGE CONSTRAINTS PER SCENARIO
        # -------------------------------------------------------------
        for s_idx, scen in enumerate(self.scenarios):
            # (a) First-stage linking: Can only allocate work if resource is contracted (z[i, t] == 1)
            # sum_j r_j * x[i, j, t, s] <= (C_i + o_i) * z[i, t]
            for r_id, res in scen.resources.items():
                for t in range(n_periods):
                    # sum_j r_j * x + idle - o == C_i * z[i, t]
                    terms_cap = {f"x_{r_id}_{t_id}_{t}_s{s_idx}": float(task.resource_demand) for t_id, task in scen.tasks.items()}
                    terms_cap[f"idle_{r_id}_{t}_s{s_idx}"] = 1.0
                    terms_cap[f"o_{r_id}_{t}_s{s_idx}"] = -1.0
                    terms_cap[f"z_{r_id}_{t}"] = -float(res.capacity_per_period)
                    self.solver.add_constraint(terms_cap, "==", 0.0, name=f"link_cap_{r_id}_{t}_s{s_idx}")

            # (b) Demand / Completion linking: sum_{i, t} x[i, j, t, s] >= d_j * y[j, s]
            for t_id, task in scen.tasks.items():
                terms_task: Dict[str, float] = {}
                for r_id in scen.resources:
                    for t in range(n_periods):
                        terms_task[f"x_{r_id}_{t_id}_{t}_s{s_idx}"] = 1.0
                terms_task[f"y_{t_id}_s{s_idx}"] = -float(task.duration)
                self.solver.add_constraint(terms_task, ">=", 0.0, name=f"task_cov_{t_id}_s{s_idx}")

        self.solver.set_objective_sense("minimize")

    def solve(
        self,
        time_limit: Optional[float] = None,
        log_to_console: bool = False
    ) -> AllocationSolution:
        """Builds and solves the two-stage stochastic extensive form."""
        self.build_model()
        res = self.solver.solve(time_limit=time_limit, log_to_console=log_to_console)

        is_feasible = res["is_feasible"]
        assignments: List[TaskAssignment] = []
        n_periods = self.base_instance.periods

        # Add back constant base penalty across all scenarios
        expected_base_penalty = sum(
            self.probabilities[s_idx] * sum(t.unmet_penalty for t in scen.tasks.values())
            for s_idx, scen in enumerate(self.scenarios)
        )
        adjusted_obj = (res["objective_value"] + expected_base_penalty) if res["objective_value"] is not None else None

        # Extract per-scenario KPIs
        if is_feasible:
            for s_idx, scen in enumerate(self.scenarios):
                scen_cost = 0.0
                unmet_count = 0
                for t_id, task in scen.tasks.items():
                    y_v = self.solver.get_var_value(f"y_{t_id}_s{s_idx}")
                    if y_v < 0.5:
                        unmet_count += 1

                self.scenario_solutions[scen.scenario_name] = {
                    "scenario_name": scen.scenario_name,
                    "probability": self.probabilities[s_idx],
                    "unmet_tasks": unmet_count,
                    "satisfaction_percent": round((len(scen.tasks) - unmet_count) / len(scen.tasks) * 100.0, 2)
                }

                # Save baseline scenario assignments as primary solution
                if s_idx == 0:
                    for r_id in scen.resources:
                        for t_id, task in scen.tasks.items():
                            for t in range(n_periods):
                                val = self.solver.get_var_value(f"x_{r_id}_{t_id}_{t}_s0")
                                if val > 0.5:
                                    assignments.append(
                                        TaskAssignment(
                                            resource_id=r_id,
                                            task_id=t_id,
                                            period=t,
                                            assigned_work=task.resource_demand
                                        )
                                    )

        kpis = {
            "expected_total_cost": round(adjusted_obj, 2) if adjusted_obj else 0.0,
            "num_scenarios_evaluated": float(self.num_scenarios),
            "baseline_assigned_tasks": float(len(assignments))
        }

        self.solution = AllocationSolution(
            instance_id=f"stochastic_{self.base_instance.instance_id}",
            status=res["status"],
            is_feasible=is_feasible,
            objective_value=adjusted_obj,
            solve_time_seconds=res["solve_time_seconds"],
            optimality_gap=0.0 if res["status"] == "OPTIMAL" else None,
            assignments=assignments,
            summary_kpis=kpis
        )
        return self.solution

