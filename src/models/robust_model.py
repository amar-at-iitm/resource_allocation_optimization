"""Robust Resource Allocation Optimization Model (Min-Max Cost / Regret)."""

from typing import Any, Dict, List, Optional
from src.data.schemas import AllocationSolution, ProblemInstance, TaskAssignment
from src.data.config_loader import ModelConfig, load_model_config
from src.optimization.solver import MILPSolver


class RobustAllocationModel:
    """Solves Robust Min-Max Cost Resource Allocation across multiple scenario realizations."""

    def __init__(
        self,
        scenarios: List[ProblemInstance],
        config: Optional[ModelConfig] = None,
        budget_gamma: Optional[int] = None
    ):
        if not scenarios:
            raise ValueError("Robust model requires at least one scenario.")
        self.scenarios = scenarios
        self.num_scenarios = len(scenarios)
        self.config = config or load_model_config()
        self.budget_gamma = budget_gamma
        self.solver = MILPSolver()
        self.base_instance = scenarios[0]
        self.solution: Optional[AllocationSolution] = None
        self.worst_case_scenario: Optional[str] = None
        self.scenario_costs: Dict[str, float] = {}

    def build_model(self) -> None:
        """Constructs Min-Max Cost formulation: min theta s.t. theta >= Cost(s) for all scenarios s."""
        resources = self.base_instance.resources
        tasks = self.base_instance.tasks
        n_periods = self.base_instance.periods
        costs = self.config.costs

        # -------------------------------------------------------------
        # 1. Epigraph variable: theta (worst-case cost)
        # -------------------------------------------------------------
        self.solver.add_variable(
            name="theta_worst_case",
            lb=0.0,
            ub=1e9,
            var_type="CONTINUOUS",
            obj_coeff=1.0
        )

        # -------------------------------------------------------------
        # 2. Master Contracting / Availability Decision z[i, t] in {0, 1}
        # -------------------------------------------------------------
        for r_id, res in resources.items():
            fixed_reservation = res.regular_cost_per_period * 0.20
            for t in range(n_periods):
                self.solver.add_variable(
                    name=f"z_{r_id}_{t}",
                    lb=0.0,
                    ub=1.0,
                    var_type="BINARY",
                    obj_coeff=0.0  # Included inside each scenario cost constraint
                )

        # -------------------------------------------------------------
        # 3. Operational Recourse Variables per Scenario s
        # -------------------------------------------------------------
        for s_idx, scen in enumerate(self.scenarios):
            # x[i, j, t, s]
            for r_id, res in scen.resources.items():
                for t_id, task in scen.tasks.items():
                    is_qualified = task.required_skill in res.skills
                    for t in range(n_periods):
                        var_name = f"x_{r_id}_{t_id}_{t}_s{s_idx}"
                        if self.config.constraints.enforce_skill_compatibility and not is_qualified:
                            self.solver.add_variable(name=var_name, lb=0.0, ub=0.0, var_type="BINARY")
                        else:
                            self.solver.add_variable(name=var_name, lb=0.0, ub=1.0, var_type="BINARY")

            # o[i, t, s]
            for r_id, res in scen.resources.items():
                max_ot = res.max_overtime_per_period if self.config.constraints.allow_overtime else 0.0
                for t in range(n_periods):
                    self.solver.add_variable(name=f"o_{r_id}_{t}_s{s_idx}", lb=0.0, ub=max_ot, var_type="CONTINUOUS")

            # idle[i, t, s]
            for r_id, res in scen.resources.items():
                for t in range(n_periods):
                    self.solver.add_variable(name=f"idle_{r_id}_{t}_s{s_idx}", lb=0.0, ub=res.capacity_per_period, var_type="CONTINUOUS")

            # y[j, s]
            for t_id in scen.tasks:
                self.solver.add_variable(name=f"y_{t_id}_s{s_idx}", lb=0.0, ub=1.0, var_type="BINARY")

        # -------------------------------------------------------------
        # 4. Constraints per Scenario s
        # -------------------------------------------------------------
        for s_idx, scen in enumerate(self.scenarios):
            # Capacity Balance linking to first-stage z[i, t]
            for r_id, res in scen.resources.items():
                for t in range(n_periods):
                    terms_cap = {f"x_{r_id}_{t_id}_{t}_s{s_idx}": float(task.resource_demand) for t_id, task in scen.tasks.items()}
                    terms_cap[f"idle_{r_id}_{t}_s{s_idx}"] = 1.0
                    terms_cap[f"o_{r_id}_{t}_s{s_idx}"] = -1.0
                    terms_cap[f"z_{r_id}_{t}"] = -float(res.capacity_per_period)
                    self.solver.add_constraint(terms_cap, "==", 0.0, name=f"rob_cap_{r_id}_{t}_s{s_idx}")

            # Task Completion
            for t_id, task in scen.tasks.items():
                terms_task: Dict[str, float] = {}
                for r_id in scen.resources:
                    for t in range(n_periods):
                        terms_task[f"x_{r_id}_{t_id}_{t}_s{s_idx}"] = 1.0
                terms_task[f"y_{t_id}_s{s_idx}"] = -float(task.duration)
                self.solver.add_constraint(terms_task, ">=", 0.0, name=f"rob_task_{t_id}_s{s_idx}")

            # Epigraph constraint: theta >= Cost(s)
            # Cost(s) = sum_z (0.2 * reg) + sum_x (hourly * dem) + sum_o (ot_cost) + sum_idle (idle_cost) + sum_unmet (pen * (1 - y))
            # Rearranged: theta - [cost terms] >= sum(penalties)
            epigraph_terms: Dict[str, float] = {"theta_worst_case": 1.0}

            # Fixed reservation cost
            for r_id, res in scen.resources.items():
                fixed_res = res.regular_cost_per_period * 0.20
                for t in range(n_periods):
                    epigraph_terms[f"z_{r_id}_{t}"] = epigraph_terms.get(f"z_{r_id}_{t}", 0.0) - fixed_res

            # Operational costs
            for r_id, res in scen.resources.items():
                hourly = res.regular_cost_per_period / max(1e-3, res.capacity_per_period)
                ot_hourly = (res.overtime_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.overtime_cost_multiplier
                idle_hourly = (res.idle_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.idle_cost_multiplier

                for t in range(n_periods):
                    epigraph_terms[f"o_{r_id}_{t}_s{s_idx}"] = -ot_hourly
                    epigraph_terms[f"idle_{r_id}_{t}_s{s_idx}"] = -idle_hourly
                    for t_id, task in scen.tasks.items():
                        x_var = f"x_{r_id}_{t_id}_{t}_s{s_idx}"
                        if x_var in self.solver._var_name_to_idx:
                            epigraph_terms[x_var] = -hourly * task.resource_demand * costs.regular_cost_multiplier

            # Unmet penalties: + penalty * (1 - y) -> theta - ... - (-pen * y) >= pen
            base_penalties = sum(t.unmet_penalty for t in scen.tasks.values())
            for t_id, task in scen.tasks.items():
                epigraph_terms[f"y_{t_id}_s{s_idx}"] = task.unmet_penalty  # subtracting cost adds positive coeff

            self.solver.add_constraint(epigraph_terms, ">=", base_penalties, name=f"epigraph_s{s_idx}")

        self.solver.set_objective_sense("minimize")

    def solve(
        self,
        time_limit: Optional[float] = None,
        log_to_console: bool = False
    ) -> AllocationSolution:
        """Solves the robust min-max problem."""
        self.build_model()
        res = self.solver.solve(time_limit=time_limit, log_to_console=log_to_console)

        is_feasible = res["is_feasible"]
        assignments: List[TaskAssignment] = []
        n_periods = self.base_instance.periods
        max_cost = -1.0
        worst_scen_name = None

        if is_feasible:
            # Evaluate actual cost under each scenario
            for s_idx, scen in enumerate(self.scenarios):
                scen_cost = 0.0
                # Reservation cost
                for r_id, res_obj in scen.resources.items():
                    fixed_res = res_obj.regular_cost_per_period * 0.20
                    for t in range(n_periods):
                        if self.solver.get_var_value(f"z_{r_id}_{t}") > 0.5:
                            scen_cost += fixed_res

                    # Overtime & Idle
                    hourly_ot = (res_obj.overtime_cost_per_period / max(1e-3, res_obj.capacity_per_period)) * self.config.costs.overtime_cost_multiplier
                    hourly_idle = (res_obj.idle_cost_per_period / max(1e-3, res_obj.capacity_per_period)) * self.config.costs.idle_cost_multiplier
                    for t in range(n_periods):
                        scen_cost += self.solver.get_var_value(f"o_{r_id}_{t}_s{s_idx}") * hourly_ot
                        scen_cost += self.solver.get_var_value(f"idle_{r_id}_{t}_s{s_idx}") * hourly_idle

                # Task assignment and unmet penalties
                for t_id, task in scen.tasks.items():
                    y_v = self.solver.get_var_value(f"y_{t_id}_s{s_idx}")
                    if y_v < 0.5:
                        scen_cost += task.unmet_penalty
                    for r_id, res_obj in scen.resources.items():
                        hourly = res_obj.regular_cost_per_period / max(1e-3, res_obj.capacity_per_period)
                        for t in range(n_periods):
                            x_v = self.solver.get_var_value(f"x_{r_id}_{t_id}_{t}_s{s_idx}")
                            if x_v > 0.5:
                                scen_cost += hourly * task.resource_demand * self.config.costs.regular_cost_multiplier

                self.scenario_costs[scen.scenario_name] = round(scen_cost, 2)
                if scen_cost > max_cost:
                    max_cost = scen_cost
                    worst_scen_name = scen.scenario_name

            self.worst_case_scenario = worst_scen_name

            # Baseline assignments
            for r_id in self.base_instance.resources:
                for t_id, task in self.base_instance.tasks.items():
                    for t in range(n_periods):
                        if self.solver.get_var_value(f"x_{r_id}_{t_id}_{t}_s0") > 0.5:
                            assignments.append(
                                TaskAssignment(
                                    resource_id=r_id,
                                    task_id=t_id,
                                    period=t,
                                    assigned_work=task.resource_demand
                                )
                            )

        kpis = {
            "worst_case_cost": round(self.solver.get_var_value("theta_worst_case"), 2) if is_feasible else 0.0,
            "worst_case_scenario": worst_scen_name or "none",
            "num_scenarios_evaluated": float(self.num_scenarios)
        }

        self.solution = AllocationSolution(
            instance_id=f"robust_{self.base_instance.instance_id}",
            status=res["status"],
            is_feasible=is_feasible,
            objective_value=res["objective_value"],
            solve_time_seconds=res["solve_time_seconds"],
            optimality_gap=0.0 if res["status"] == "OPTIMAL" else None,
            assignments=assignments,
            summary_kpis=kpis
        )
        return self.solution

