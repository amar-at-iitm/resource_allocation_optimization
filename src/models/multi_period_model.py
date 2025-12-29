"""Multi-Period Resource Allocation and Scheduling Optimization Model using HiGHS."""

from typing import Any, Dict, List, Optional
from src.data.schemas import AllocationSolution, ProblemInstance, TaskAssignment
from src.data.config_loader import ModelConfig, load_model_config
from src.optimization.solver import MILPSolver


class MultiPeriodSchedulingModel:
    """Solves multi-period resource allocation and scheduling with precedence constraints."""

    def __init__(self, instance: ProblemInstance, config: Optional[ModelConfig] = None):
        self.instance = instance
        self.config = config or load_model_config()
        self.solver = MILPSolver()
        self.solution: Optional[AllocationSolution] = None

    def build_model(self) -> None:
        """Constructs variables, precedence constraints, capacity balances, and multi-period objective."""
        resources = self.instance.resources
        tasks = self.instance.tasks
        n_periods = self.instance.periods
        costs = self.config.costs

        # 1. Decision Variables
        # x[i, j, t]: Resource i assigned to task j at period t
        for r_id, res in resources.items():
            for t_id, task in tasks.items():
                is_qualified = task.required_skill in res.skills
                for t in range(n_periods):
                    var_name = f"x_{r_id}_{t_id}_{t}"
                    if self.config.constraints.enforce_skill_compatibility and not is_qualified:
                        self.solver.add_variable(name=var_name, lb=0.0, ub=0.0, var_type="BINARY", obj_coeff=0.0)
                    else:
                        hourly_rate = res.regular_cost_per_period / max(1e-3, res.capacity_per_period)
                        reg_cost = hourly_rate * costs.regular_cost_multiplier * task.resource_demand
                        self.solver.add_variable(name=var_name, lb=0.0, ub=1.0, var_type="BINARY", obj_coeff=reg_cost)

        # s[j, t]: Task j starts at period t
        for t_id, task in tasks.items():
            max_start = n_periods - task.duration
            for t in range(n_periods):
                var_name = f"s_{t_id}_{t}"
                if t < task.release_date or t > max_start:
                    self.solver.add_variable(name=var_name, lb=0.0, ub=0.0, var_type="BINARY", obj_coeff=0.0)
                else:
                    self.solver.add_variable(name=var_name, lb=0.0, ub=1.0, var_type="BINARY", obj_coeff=0.0)

        # y[j]: Task j is completed (1) or unmet (0)
        for t_id, task in tasks.items():
            # Penalty when y_j = 0 is equivalent to -penalty * y_j + const
            self.solver.add_variable(
                name=f"y_{t_id}",
                lb=0.0,
                ub=1.0,
                var_type="BINARY",
                obj_coeff=-task.unmet_penalty
            )

        # o[i, t]: Overtime for resource i at period t
        for r_id, res in resources.items():
            max_ot = res.max_overtime_per_period if self.config.constraints.allow_overtime else 0.0
            ot_cost = (res.overtime_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.overtime_cost_multiplier
            for t in range(n_periods):
                self.solver.add_variable(
                    name=f"o_{r_id}_{t}",
                    lb=0.0,
                    ub=max_ot,
                    var_type="CONTINUOUS",
                    obj_coeff=ot_cost
                )

        # idle[i, t]: Idle capacity for resource i at period t
        for r_id, res in resources.items():
            idle_cost = (res.idle_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.idle_cost_multiplier
            for t in range(n_periods):
                self.solver.add_variable(
                    name=f"idle_{r_id}_{t}",
                    lb=0.0,
                    ub=res.capacity_per_period,
                    var_type="CONTINUOUS",
                    obj_coeff=idle_cost
                )

        # 2. Constraints

        # (a) Start and Completion consistency: sum_{t} s[j, t] == y[j]
        for t_id, task in tasks.items():
            max_start = n_periods - task.duration
            terms: Dict[str, float] = {}
            for t in range(task.release_date, max_start + 1):
                if t < n_periods:
                    terms[f"s_{t_id}_{t}"] = 1.0
            terms[f"y_{t_id}"] = -1.0
            self.solver.add_constraint(terms, "==", 0.0, name=f"start_sum_{t_id}")

        # (b) Duration and continuity: for start period t, execution runs for d_j periods
        for t_id, task in tasks.items():
            max_start = n_periods - task.duration
            for t in range(task.release_date, max_start + 1):
                for tau in range(task.duration):
                    act_t = t + tau
                    terms = {f"x_{r_id}_{t_id}_{act_t}": 1.0 for r_id in resources}
                    terms[f"s_{t_id}_{t}"] = -1.0
                    self.solver.add_constraint(terms, ">=", 0.0, name=f"cont_{t_id}_{t}_{tau}")

            # Total execution periods == d_j * y[j]
            tot_terms: Dict[str, float] = {}
            for r_id in resources:
                for t in range(n_periods):
                    tot_terms[f"x_{r_id}_{t_id}_{t}"] = 1.0
            tot_terms[f"y_{t_id}"] = -float(task.duration)
            self.solver.add_constraint(tot_terms, "==", 0.0, name=f"tot_dur_{t_id}")

            # At most one resource per task at any period t
            for t in range(n_periods):
                res_terms = {f"x_{r_id}_{t_id}_{t}": 1.0 for r_id in resources}
                self.solver.add_constraint(res_terms, "<=", 1.0, name=f"single_res_{t_id}_{t}")

        # (c) Precedence Constraints
        if self.config.constraints.enforce_precedence:
            for t_id, task in tasks.items():
                for pred_id in task.predecessors:
                    if pred_id not in tasks:
                        continue
                    pred_task = tasks[pred_id]
                    # If pred is not completed, succ cannot be completed
                    self.solver.add_constraint(
                        {f"y_{t_id}": 1.0, f"y_{pred_id}": -1.0},
                        "<=",
                        0.0,
                        name=f"prec_y_{pred_id}_{t_id}"
                    )
                    # Task t_id starting at t implies pred_id started at or before t - d_k
                    for t in range(n_periods):
                        if f"s_{t_id}_{t}" in self.solver._var_name_to_idx:
                            # sum_{t' <= t - d_k} s[pred, t'] >= s[task, t]
                            valid_pred_starts = [
                                t_prime for t_prime in range(n_periods)
                                if t_prime <= t - pred_task.duration
                            ]
                            terms_prec = {f"s_{pred_id}_{tp}": 1.0 for tp in valid_pred_starts}
                            terms_prec[f"s_{t_id}_{t}"] = -1.0
                            self.solver.add_constraint(terms_prec, ">=", 0.0, name=f"prec_{pred_id}_{t_id}_{t}")

        # (d) Capacity & Overtime Balance per Resource and Period
        for r_id, res in resources.items():
            for t in range(n_periods):
                terms_cap = {f"x_{r_id}_{t_id}_{t}": float(task.resource_demand) for t_id, task in tasks.items()}
                terms_cap[f"idle_{r_id}_{t}"] = 1.0
                terms_cap[f"o_{r_id}_{t}"] = -1.0
                self.solver.add_constraint(terms_cap, "==", float(res.capacity_per_period), name=f"cap_{r_id}_{t}")

        self.solver.set_objective_sense("minimize")

    def solve(
        self,
        time_limit: Optional[float] = None,
        log_to_console: bool = False
    ) -> AllocationSolution:
        """Builds and solves the multi-period optimization model."""
        self.build_model()
        res = self.solver.solve(time_limit=time_limit, log_to_console=log_to_console)

        assignments: List[TaskAssignment] = []
        overtime_dict: Dict[str, Dict[int, float]] = {}
        idle_dict: Dict[str, Dict[int, float]] = {}
        unmet_dict: Dict[str, Dict[int, float]] = {}
        start_times: Dict[str, int] = {}
        completion_times: Dict[str, int] = {}

        is_feasible = res["is_feasible"]
        total_assigned_demand = 0.0
        total_overtime_hours = 0.0
        total_idle_hours = 0.0
        unmet_tasks_count = 0
        n_periods = self.instance.periods

        # Adjust objective value by adding back the sum of all task penalties
        # because we used: obj_coeff = -unmet_penalty * y_j
        # Actual cost = sum(...) + sum_j penalty * (1 - y_j)
        base_penalties = sum(t.unmet_penalty for t in self.instance.tasks.values())
        adjusted_obj_value = (res["objective_value"] + base_penalties) if res["objective_value"] is not None else None

        if is_feasible:
            # Overtime & Idle
            for r_id in self.instance.resources:
                overtime_dict[r_id] = {}
                idle_dict[r_id] = {}
                for t in range(n_periods):
                    ot_val = round(self.solver.get_var_value(f"o_{r_id}_{t}"), 3)
                    idle_val = round(self.solver.get_var_value(f"idle_{r_id}_{t}"), 3)
                    overtime_dict[r_id][t] = ot_val
                    idle_dict[r_id][t] = idle_val
                    total_overtime_hours += ot_val
                    total_idle_hours += idle_val

            # Assignments
            for r_id in self.instance.resources:
                for t_id, task in self.instance.tasks.items():
                    for t in range(n_periods):
                        val = self.solver.get_var_value(f"x_{r_id}_{t_id}_{t}")
                        if val > 0.5:
                            assignments.append(
                                TaskAssignment(
                                    resource_id=r_id,
                                    task_id=t_id,
                                    period=t,
                                    assigned_work=task.resource_demand
                                )
                            )
                            total_assigned_demand += task.resource_demand

            # Start and completion times
            for t_id, task in self.instance.tasks.items():
                y_val = self.solver.get_var_value(f"y_{t_id}")
                if y_val > 0.5:
                    for t in range(n_periods):
                        s_val = self.solver.get_var_value(f"s_{t_id}_{t}")
                        if s_val > 0.5:
                            start_times[t_id] = t
                            completion_times[t_id] = t + task.duration - 1
                            break
                    unmet_dict[t_id] = {0: 0.0}
                else:
                    unmet_tasks_count += 1
                    unmet_dict[t_id] = {0: 1.0}

        total_system_capacity = sum(r.capacity_per_period for r in self.instance.resources.values()) * n_periods
        utilization_rate = round((total_assigned_demand / total_system_capacity * 100.0), 2) if total_system_capacity > 0 else 0.0
        makespan = max(completion_times.values()) + 1 if completion_times else 0

        kpis = {
            "total_assigned_tasks": float(len(self.instance.tasks) - unmet_tasks_count),
            "total_unmet_tasks": float(unmet_tasks_count),
            "total_overtime_hours": round(total_overtime_hours, 2),
            "total_idle_hours": round(total_idle_hours, 2),
            "capacity_utilization_percent": utilization_rate,
            "demand_satisfaction_percent": round(
                (len(self.instance.tasks) - unmet_tasks_count) / max(1, len(self.instance.tasks)) * 100.0, 2
            ),
            "makespan_periods": float(makespan)
        }

        self.solution = AllocationSolution(
            instance_id=self.instance.instance_id,
            status=res["status"],
            is_feasible=is_feasible,
            objective_value=adjusted_obj_value,
            solve_time_seconds=res["solve_time_seconds"],
            optimality_gap=0.0 if res["status"] == "OPTIMAL" else None,
            assignments=assignments,
            overtime=overtime_dict,
            idle=idle_dict,
            unmet_demand=unmet_dict,
            task_start_times=start_times,
            task_completion_times=completion_times,
            summary_kpis=kpis
        )
        return self.solution

