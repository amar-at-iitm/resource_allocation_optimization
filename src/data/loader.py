"""Benchmark loader and PSPLIB / RCPSP parser mapping into ProblemInstance."""

from pathlib import Path
from typing import Dict, List, Optional
import json

from src.data.schemas import ProblemInstance, Resource, Task


def parse_psplib_file(file_path: Path | str) -> ProblemInstance:
    """Parses a standard PSPLIB / RCPSP benchmark file (.sm format)
    into the project's standardized ProblemInstance schema.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    instance_id = path.stem
    section = None

    due_date = 30
    precedence_successors: Dict[int, List[int]] = {}
    durations: Dict[int, int] = {}
    resource_demands: Dict[int, List[float]] = {}
    resource_capacities: List[float] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("*"):
            i += 1
            continue

        if "PROJECT INFORMATION:" in line:
            section = "PROJECT"
            i += 1
            continue
        elif "PRECEDENCE RELATIONS:" in line:
            section = "PRECEDENCE"
            i += 1
            continue
        elif "REQUESTS/DURATIONS:" in line:
            section = "REQUESTS"
            i += 1
            continue
        elif "RESOURCEAVAILABILITIES:" in line:
            section = "AVAILABILITY"
            i += 1
            continue

        if section == "PROJECT":
            if line.startswith("pronr."):
                i += 1
                data_line = lines[i].split()
                if len(data_line) >= 4:
                    due_date = int(data_line[3])
            i += 1
            continue

        elif section == "PRECEDENCE":
            if line.startswith("jobnr."):
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 3:
                job_nr = int(parts[0])
                num_succ = int(parts[2])
                succs = [int(p) for p in parts[3:3 + num_succ]]
                precedence_successors[job_nr] = succs
            i += 1
            continue

        elif section == "REQUESTS":
            if line.startswith("jobnr.") or line.startswith("---"):
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 3:
                job_nr = int(parts[0])
                # parts[1] is mode
                dur = int(parts[2])
                reqs = [float(r) for r in parts[3:]]
                durations[job_nr] = dur
                resource_demands[job_nr] = reqs
            i += 1
            continue

        elif section == "AVAILABILITY":
            if line.startswith("R"):
                i += 1
                continue
            parts = line.split()
            if parts:
                resource_capacities = [float(p) for p in parts]
                section = None
            i += 1
            continue

        i += 1

    # Invert successors to get predecessors
    all_jobs = sorted(durations.keys())
    predecessors_map: Dict[int, List[int]] = {j: [] for j in all_jobs}
    for job, succs in precedence_successors.items():
        for succ in succs:
            if succ in predecessors_map:
                predecessors_map[succ].append(job)

    # Filter out dummy start (1) and dummy end (max job) if they have duration 0
    # or keep them as boundary tasks. We include non-dummy tasks:
    real_jobs = [j for j in all_jobs if durations[j] > 0]
    if not real_jobs:  # Fallback if all are 0
        real_jobs = all_jobs

    # 1. Build Resources (one per resource type or scaled capacity)
    num_res_types = len(resource_capacities) if resource_capacities else 4
    if not resource_capacities:
        resource_capacities = [8.0] * num_res_types

    resources: Dict[str, Resource] = {}
    for idx, cap in enumerate(resource_capacities):
        r_id = f"R_{idx+1:03d}"
        skill_name = f"Skill_R{idx+1}"
        resources[r_id] = Resource(
            id=r_id,
            name=f"Resource_Type_{idx+1}",
            skills=[skill_name],
            capacity_per_period=cap,
            regular_cost_per_period=100.0 + idx * 10.0,
            overtime_cost_per_period=(100.0 + idx * 10.0) * 1.5,
            max_overtime_per_period=round(cap * 0.25, 2),
            idle_cost_per_period=25.0
        )

    # 2. Build Tasks
    tasks: Dict[str, Task] = {}
    for job_id in real_jobs:
        t_id = f"T_{job_id:03d}"
        dur = max(1, durations.get(job_id, 1))
        reqs = resource_demands.get(job_id, [1.0])

        # Pick primary skill as the resource type with highest demand
        if reqs and max(reqs) > 0:
            max_req_idx = int(reqs.index(max(reqs)))
            primary_skill = f"Skill_R{max_req_idx+1}"
            demand_val = float(reqs[max_req_idx])
        else:
            primary_skill = "Skill_R1"
            demand_val = 1.0

        # Map predecessors to only real jobs
        preds = [f"T_{p:03d}" for p in predecessors_map.get(job_id, []) if p in real_jobs]

        tasks[t_id] = Task(
            id=t_id,
            name=f"Job_{job_id}",
            required_skill=primary_skill,
            duration=dur,
            resource_demand=max(0.5, demand_val),
            priority=2.0,
            unmet_penalty=1000.0,
            predecessors=preds,
            release_date=0,
            due_date=due_date
        )

    planning_horizon = max(due_date, sum(durations.get(j, 1) for j in real_jobs))

    return ProblemInstance(
        instance_id=instance_id,
        periods=min(planning_horizon, 40),
        resources=resources,
        tasks=tasks,
        scenario_name="psplib_benchmark",
        metadata={"raw_file": str(path), "total_jobs_in_file": len(all_jobs)}
    )


def load_benchmark(file_path: Path | str) -> ProblemInstance:
    """Loads a benchmark instance (.sm or .json)."""
    path = Path(file_path)
    if path.suffix == ".json":
        return ProblemInstance.load_json(path)
    elif path.suffix in [".sm", ".rcp"]:
        return parse_psplib_file(path)
    else:
        # Default try parsing as PSPLIB format
        return parse_psplib_file(path)


def load_instance_json(file_path: Path | str) -> ProblemInstance:
    """Helper to load a JSON ProblemInstance file."""
    return ProblemInstance.load_json(file_path)

