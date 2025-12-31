"""Single-Period Deterministic Resource Allocation Model using HiGHS."""

from typing import Any, Dict, List, Optional
from src.data.schemas import AllocationSolution, ProblemInstance, TaskAssignment
from src.data.config_loader import ModelConfig, load_model_config
from src.optimization.solver import MILPSolver


class DeterministicAllocationModel:
    """Solves baseline single-period resource allocation problem."""

    def __init__(self, instance: ProblemInstance, config: Optional[ModelConfig] = None):
        self.instance = instance
        self.config = config or load_model_config()
        self.solver = MILPSolver()
        self.solution: Optional[AllocationSolution] = None

    def build_model(self) -> None:
        """Constructs variables, constraints, and objective for single-period allocation."""
        resources = self.instance.resources
        tasks = self.instance.tasks
        costs = self.config.costs

        # 1. Variables
        # x[i, j]: Binary assignment
        for r_id, res in resources.items():
            for t_id, task in tasks.items():
                # Check skill qualification
                is_qualified = task.required_skill in res.skills
                if self.config.constraints.enforce_skill_compatibility and not is_qualified:
                    # Fix to 0
                    self.solver.add_variable(
                        name=f"x_{r_id}_{t_id}",
                        lb=0.0,
                        ub=0.0,
                        var_type="BINARY",
                        obj_coeff=0.0
                    )
                else:
                    reg_unit_cost = res.regular_cost_per_period * costs.regular_cost_multiplier
                    # Proportional cost for task duration and demand
                    assign_cost = reg_unit_cost * task.resource_demand
                    self.solver.add_variable(
                        name=f"x_{r_id}_{t_id}",
                        lb=0.0,
                        ub=1.0,
                        var_type="BINARY",
                        obj_coeff=assign_cost
                    )

        # o[i]: Overtime
        for r_id, res in resources.items():
            max_ot = res.max_overtime_per_period if self.config.constraints.allow_overtime else 0.0
            ot_cost = res.overtime_cost_per_period * costs.overtime_cost_multiplier
            self.solver.add_variable(
                name=f"o_{r_id}",
                lb=0.0,
                ub=max_ot,
                var_type="CONTINUOUS",
                obj_coeff=ot_cost
            )

        # idle[i]: Idle capacity
        for r_id, res in resources.items():
            idle_cost = res.idle_cost_per_period * costs.idle_cost_multiplier
            self.solver.add_variable(
                name=f"idle_{r_id}",
                lb=0.0,
                ub=res.capacity_per_period,
                var_type="CONTINUOUS",
                obj_coeff=idle_cost
            )

        # u[j]: Unmet demand penalty
        for t_id, task in tasks.items():
            penalty = task.unmet_penalty
            self.solver.add_variable(
                name=f"u_{t_id}",
                lb=0.0,
                ub=1.0,
                var_type="CONTINUOUS",
                obj_coeff=penalty
            )

        # 2. Constraints
        # Demand Satisfaction: sum_i x[i, j] + u[j] == 1
        for t_id in tasks:
            terms: Dict[str, float] = {f"x_{r_id}_{t_id}": 1.0 for r_id in resources}
            terms[f"u_{t_id}"] = 1.0
            self.solver.add_constraint(terms, "==", 1.0, name=f"demand_{t_id}")

        # Resource Capacity Balance: sum_j r_j * x[i, j] + idle[i] - o[i] == C_i
        for r_id, res in resources.items():
            terms = {f"x_{r_id}_{t_id}": float(task.resource_demand) for t_id, task in tasks.items()}
            terms[f"idle_{r_id}"] = 1.0
            terms[f"o_{r_id}"] = -1.0
            self.solver.add_constraint(terms, "==", float(res.capacity_per_period), name=f"cap_{r_id}")

        self.solver.set_objective_sense("minimize")

    def solve(
        self,
        time_limit: Optional[float] = None,
        log_to_console: bool = False
    ) -> AllocationSolution:
        """Builds and solves the model, returning structured AllocationSolution."""
        self.build_model()
        res = self.solver.solve(time_limit=time_limit, log_to_console=log_to_console)

        assignments: List[TaskAssignment] = []
        overtime_dict: Dict[str, Dict[int, float]] = {}
        idle_dict: Dict[str, Dict[int, float]] = {}
        unmet_dict: Dict[str, Dict[int, float]] = {}

        is_feasible = res["is_feasible"]
        total_assigned_demand = 0.0
        total_overtime_hours = 0.0
        total_idle_hours = 0.0
        unmet_tasks_count = 0

        if is_feasible:
            # Extract assignments
            for r_id, res_obj in self.instance.resources.items():
                ot_val = round(self.solver.get_var_value(f"o_{r_id}"), 3)
                idle_val = round(self.solver.get_var_value(f"idle_{r_id}"), 3)
                overtime_dict[r_id] = {0: ot_val}
                idle_dict[r_id] = {0: idle_val}
                total_overtime_hours += ot_val
                total_idle_hours += idle_val

                for t_id, task in self.instance.tasks.items():
                    val = self.solver.get_var_value(f"x_{r_id}_{t_id}")
                    if val > 0.5:
                        assignments.append(
                            TaskAssignment(
                                resource_id=r_id,
                                task_id=t_id,
                                period=0,
                                assigned_work=task.resource_demand
                            )
                        )
                        total_assigned_demand += task.resource_demand

            # Extract unmet demand
            for t_id in self.instance.tasks:
                u_val = round(self.solver.get_var_value(f"u_{t_id}"), 3)
                unmet_dict[t_id] = {0: u_val}
                if u_val > 0.5:
                    unmet_tasks_count += 1

        total_capacity = sum(r.capacity_per_period for r in self.instance.resources.values())
        utilization_rate = round((total_assigned_demand / total_capacity * 100.0), 2) if total_capacity > 0 else 0.0

        kpis = {
            "total_assigned_tasks": float(len(assignments)),
            "total_unmet_tasks": float(unmet_tasks_count),
            "total_overtime_hours": round(total_overtime_hours, 2),
            "total_idle_hours": round(total_idle_hours, 2),
            "capacity_utilization_percent": utilization_rate,
            "demand_satisfaction_percent": round(
                (len(self.instance.tasks) - unmet_tasks_count) / max(1, len(self.instance.tasks)) * 100.0, 2
            )
        }

        self.solution = AllocationSolution(
            instance_id=self.instance.instance_id,
            status=res["status"],
            is_feasible=is_feasible,
            objective_value=res["objective_value"],
            solve_time_seconds=res["solve_time_seconds"],
            optimality_gap=0.0 if res["status"] == "OPTIMAL" else None,
            assignments=assignments,
            overtime=overtime_dict,
            idle=idle_dict,
            unmet_demand=unmet_dict,
            summary_kpis=kpis
        )
        return self.solution

