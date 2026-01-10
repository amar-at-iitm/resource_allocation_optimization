"""Chromosome representation and Serial Schedule Generation Scheme (SGS) decoder."""

from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance, TaskAssignment
from src.data.config_loader import ModelConfig, load_model_config
from src.data.validation import is_dag


class ScheduleChromosome:
    """Individual representation encoding task execution priority order and resource assignment."""

    def __init__(self, task_order: List[str], resource_map: Dict[str, str]):
        self.task_order = list(task_order)
        self.resource_map = dict(resource_map)

    def copy(self) -> "ScheduleChromosome":
        return ScheduleChromosome(self.task_order.copy(), self.resource_map.copy())

    def repair_precedence(self, instance: ProblemInstance) -> None:
        """Repairs task_order so that it forms a strictly valid topological sort."""
        tasks = instance.tasks
        in_degrees: Dict[str, int] = {t_id: 0 for t_id in tasks}
        adjacency: Dict[str, List[str]] = {t_id: [] for t_id in tasks}

        for t_id, task in tasks.items():
            for pred in task.predecessors:
                if pred in tasks:
                    adjacency[pred].append(t_id)
                    in_degrees[t_id] += 1

        # Priority dictated by position in current task_order
        priority = {t_id: idx for idx, t_id in enumerate(self.task_order)}

        # Kahn's algorithm picking available task with lowest priority index
        available = [t_id for t_id, deg in in_degrees.items() if deg == 0]
        repaired_order: List[str] = []

        while available:
            available.sort(key=lambda t: priority.get(t, 0))
            curr = available.pop(0)
            repaired_order.append(curr)
            for neighbor in adjacency[curr]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    available.append(neighbor)

        if len(repaired_order) == len(tasks):
            self.task_order = repaired_order


def generate_random_chromosome(
    instance: ProblemInstance,
    rng: np.random.Generator
) -> ScheduleChromosome:
    """Creates a random valid chromosome satisfying precedence and skill qualifications."""
    # 1. Topological order with randomized tie-breaking
    valid, base_topo = is_dag(instance.tasks)
    if not valid:
        base_topo = list(instance.tasks.keys())

    # Random permutation of tasks repaired via topological sort
    task_keys = list(instance.tasks.keys())
    rng.shuffle(task_keys)

    # 2. Resource mapping choosing from qualified resources
    skill_coverage = instance.get_skill_coverage()
    resource_map: Dict[str, str] = {}

    for t_id, task in instance.tasks.items():
        qualified = skill_coverage.get(task.required_skill, [])
        if qualified:
            resource_map[t_id] = str(rng.choice(qualified))
        else:
            # Fallback to any resource if skill missing
            resource_map[t_id] = str(rng.choice(list(instance.resources.keys())))

    chrom = ScheduleChromosome(task_order=task_keys, resource_map=resource_map)
    chrom.repair_precedence(instance)
    return chrom


class ScheduleDecoder:
    """Serial Schedule Generation Scheme (SGS) decoding chromosomes into feasible solutions."""

    def __init__(self, instance: ProblemInstance, config: Optional[ModelConfig] = None):
        self.instance = instance
        self.config = config or load_model_config()

    def decode(self, chromosome: ScheduleChromosome) -> AllocationSolution:
        """Decodes chromosome into a complete AllocationSolution and calculates operational costs."""
        n_periods = self.instance.periods
        resources = self.instance.resources
        tasks = self.instance.tasks
        costs = self.config.costs

        # Track remaining regular capacity and overtime per resource and period
        # res_cap_used[r_id][t]
        res_workload: Dict[str, Dict[int, float]] = {
            r_id: {t: 0.0 for t in range(n_periods)}
            for r_id in resources
        }

        task_start_times: Dict[str, int] = {}
        task_completion_times: Dict[str, int] = {}
        assignments: List[TaskAssignment] = []
        unmet_demand: Dict[str, Dict[int, float]] = {}

        # Decode tasks in the sequence specified by the chromosome
        for t_id in chromosome.task_order:
            task = tasks[t_id]
            assigned_r_id = chromosome.resource_map.get(t_id)

            if not assigned_r_id or assigned_r_id not in resources:
                unmet_demand[t_id] = {0: 1.0}
                continue

            res = resources[assigned_r_id]

            # Verify skill qualification
            if self.config.constraints.enforce_skill_compatibility and task.required_skill not in res.skills:
                unmet_demand[t_id] = {0: 1.0}
                continue

            # Earliest start period based on release date and predecessor completions
            earliest_start = task.release_date
            for pred_id in task.predecessors:
                if pred_id in task_completion_times:
                    earliest_start = max(earliest_start, task_completion_times[pred_id] + 1)
                else:
                    # Predecessor was unmet/not scheduled -> current task cannot be scheduled
                    earliest_start = n_periods + 1
                    break

            if earliest_start + task.duration > n_periods:
                unmet_demand[t_id] = {0: 1.0}
                continue

            # Find earliest period t >= earliest_start with available capacity for all duration periods
            scheduled_start: Optional[int] = None
            max_start = n_periods - task.duration

            for t in range(earliest_start, max_start + 1):
                can_fit = True
                for tau in range(task.duration):
                    curr_t = t + tau
                    curr_load = res_workload[assigned_r_id][curr_t]
                    max_cap = res.capacity_per_period + (res.max_overtime_per_period if self.config.constraints.allow_overtime else 0.0)
                    if curr_load + task.resource_demand > max_cap + 1e-4:
                        can_fit = False
                        break
                if can_fit:
                    scheduled_start = t
                    break

            if scheduled_start is not None:
                # Schedule task
                task_start_times[t_id] = scheduled_start
                task_completion_times[t_id] = scheduled_start + task.duration - 1
                unmet_demand[t_id] = {0: 0.0}

                for tau in range(task.duration):
                    period_t = scheduled_start + tau
                    res_workload[assigned_r_id][period_t] += task.resource_demand
                    assignments.append(
                        TaskAssignment(
                            resource_id=assigned_r_id,
                            task_id=t_id,
                            period=period_t,
                            assigned_work=task.resource_demand
                        )
                    )
            else:
                unmet_demand[t_id] = {0: 1.0}

        # Compute Overtime, Idle, and Costs
        total_cost = 0.0
        overtime_dict: Dict[str, Dict[int, float]] = {r_id: {} for r_id in resources}
        idle_dict: Dict[str, Dict[int, float]] = {r_id: {} for r_id in resources}
        total_overtime_hours = 0.0
        total_idle_hours = 0.0

        for r_id, res in resources.items():
            hourly_reg = (res.regular_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.regular_cost_multiplier
            hourly_ot = (res.overtime_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.overtime_cost_multiplier
            hourly_idle = (res.idle_cost_per_period / max(1e-3, res.capacity_per_period)) * costs.idle_cost_multiplier

            for t in range(n_periods):
                workload = res_workload[r_id][t]
                if workload <= res.capacity_per_period:
                    reg_work = workload
                    ot_work = 0.0
                    idle_work = res.capacity_per_period - workload
                else:
                    reg_work = res.capacity_per_period
                    ot_work = workload - res.capacity_per_period
                    idle_work = 0.0

                overtime_dict[r_id][t] = round(ot_work, 3)
                idle_dict[r_id][t] = round(idle_work, 3)
                total_overtime_hours += ot_work
                total_idle_hours += idle_work

                total_cost += (reg_work * hourly_reg) + (ot_work * hourly_ot) + (idle_work * hourly_idle)

        # Unmet demand penalties
        unmet_count = 0
        for t_id, task in tasks.items():
            if t_id not in task_start_times:
                total_cost += task.unmet_penalty
                unmet_count += 1

        total_system_capacity = sum(r.capacity_per_period for r in resources.values()) * n_periods
        total_assigned_work = sum(a.assigned_work for a in assignments)
        utilization = round((total_assigned_work / total_system_capacity * 100.0), 2) if total_system_capacity > 0 else 0.0
        makespan = max(task_completion_times.values()) + 1 if task_completion_times else 0

        kpis = {
            "total_assigned_tasks": float(len(tasks) - unmet_count),
            "total_unmet_tasks": float(unmet_count),
            "total_overtime_hours": round(total_overtime_hours, 2),
            "total_idle_hours": round(total_idle_hours, 2),
            "capacity_utilization_percent": utilization,
            "demand_satisfaction_percent": round((len(tasks) - unmet_count) / max(1, len(tasks)) * 100.0, 2),
            "makespan_periods": float(makespan)
        }

        return AllocationSolution(
            instance_id=self.instance.instance_id,
            status="FEASIBLE",
            is_feasible=True,
            objective_value=round(total_cost, 2),
            solve_time_seconds=0.0,
            optimality_gap=None,
            assignments=assignments,
            overtime=overtime_dict,
            idle=idle_dict,
            unmet_demand=unmet_demand,
            task_start_times=task_start_times,
            task_completion_times=task_completion_times,
            summary_kpis=kpis
        )

