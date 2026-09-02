# Multi-Period Resource Allocation and Scheduling under Uncertainty

> An Operations Research decision-support framework for optimally allocating limited resources across tasks and time periods under capacity, cost, and uncertainty constraints.

## Overview

Organizations frequently operate with limited resources while facing uncertain demand and changing operational requirements.

Examples include:

* Allocating employees to work shifts.
* Assigning machines to production tasks.
* Scheduling hospital staff.
* Allocating service capacity.
* Assigning project resources.
* Managing call-center personnel.

Although the application domains differ, these problems share a common mathematical structure:

> **Limited resources must be allocated to competing tasks over time while minimizing cost and satisfying operational requirements.**

This project develops a mathematical optimization framework for solving multi-period resource allocation and scheduling problems.

The framework integrates:

```text
Mathematical Modelling
+
Mixed-Integer Optimization
+
Scheduling
+
Uncertainty Modelling
+
Scenario Analysis
+
Metaheuristic Optimization
```

The project is primarily intended for applications in:

* Operations Research
* Optimization
* Decision Science
* Industrial Engineering
* Operations Analytics
* Resource Planning
* Workforce Optimization

---

# Business Problem

Suppose an organization has:

* Multiple resources.
* Multiple tasks.
* Multiple planning periods.
* Different resource capabilities.
* Limited capacity.
* Time-dependent demand.
* Different operational costs.

The organization must decide:

```text
Which resource
should perform which task
during which time period?
```

The decision must satisfy operational requirements while minimizing total cost.

---

# General Mathematical Structure

The project considers:

```text
Resources
R = {1, 2, ..., I}

Tasks
J = {1, 2, ..., J}

Time Periods
T = {1, 2, ..., T}
```

The resource allocation decisions can represent:

* Employees
* Machines
* Vehicles
* Teams
* Processing units

The tasks can represent:

* Production jobs
* Customer demand
* Service requests
* Projects
* Operational activities

---

# Project Objectives

The objectives are:

1. Formulate a real-world allocation problem mathematically.
2. Develop a Mixed-Integer Linear Programming model.
3. Implement deterministic resource allocation.
4. Introduce uncertainty into demand or resource availability.
5. Perform scenario analysis.
6. Compare exact optimization and heuristic methods.
7. Evaluate scalability and computational performance.
8. Develop visualizations for decision support.

---

# Decision Variables

A primary binary decision variable may be defined as:

```text
x[i,j,t] = 1
```

if resource `i` is assigned to task `j` during period `t`.

Otherwise:

```text
x[i,j,t] = 0
```

Additional decision variables may include:

```text
o[i,t] → overtime assigned to resource i

u[j,t] → unmet demand for task j

y[i,t] → resource availability

z[i,j,t] → task-resource assignment
```

---

# Objective Function

The baseline model minimizes total operational cost.

```text
Total Cost
=
Regular Resource Cost
+
Overtime Cost
+
Idle Resource Cost
+
Unmet Demand Penalty
```

The general optimization problem can therefore be expressed as:

```text
Minimize:

Σ Regular Assignment Costs
+
Σ Overtime Costs
+
Σ Idle Costs
+
Σ Unmet Demand Penalties
```

---

# Constraints

## 1. Resource Availability

A resource cannot be assigned if unavailable.

---

## 2. Capacity Constraint

A resource has limited capacity during each planning period.

```text
Assigned Work
≤
Available Capacity
```

---

## 3. Demand Satisfaction

The assigned resources must satisfy operational demand.

When complete satisfaction is impossible, unmet demand may be penalized.

---

## 4. Skill Compatibility

Certain resources may only perform specific tasks.

```text
Resource Skill Level
≥
Task Requirement
```

---

## 5. Working-Hour Constraint

Employees cannot exceed maximum permitted working hours.

---

## 6. Overtime Constraint

Overtime is limited.

```text
Overtime
≤
Maximum Allowed Overtime
```

---

## 7. Scheduling Constraints

Depending on the application, the model may include:

* Minimum rest periods.
* Maximum consecutive working periods.
* Shift restrictions.
* Task precedence relationships.

---

# Model Variants

The project should not consist of only one optimization model.

Instead, it should progressively develop multiple versions.

## Model 1 — Deterministic Resource Allocation

Assume:

```text
Demand is known.
Resource availability is known.
Task duration is known.
```

This establishes the baseline optimization model.

---

## Model 2 — Multi-Period Scheduling

Extend the model to include multiple time periods.

Example:

```text
Day 1
Day 2
Day 3
...
Day T
```

Resources must now be allocated while considering future demand and capacity.

---

## Model 3 — Demand Uncertainty

Demand is no longer assumed to be perfectly known.

Generate scenarios such as:

```text
Low Demand
Normal Demand
High Demand
Peak Demand
```

---

## Model 4 — Resource Availability Uncertainty

Model:

* Employee absenteeism.
* Machine failure.
* Unexpected resource downtime.

---

## Model 5 — Robust Optimization

Develop solutions that remain feasible or cost-effective under adverse scenarios.

---

# Exact Optimization

The primary exact approach should use:

* Pyomo
* PuLP
* OR-Tools

The model can be solved using available solvers such as:

* HiGHS
* CBC
* GLPK
* Gurobi

The implementation should separate:

```text
Data
Model Definition
Objective
Constraints
Solver Configuration
Results Extraction
```

This makes the code reusable and easier to modify.

---

# Metaheuristic Optimization

To make the project more technically interesting, implement at least one heuristic or metaheuristic approach.

Recommended options:

## Genetic Algorithm

Suitable for large or complex allocation problems.

Typical workflow:

```text
Initialize Population
        │
        ▼
Evaluate Fitness
        │
        ▼
Selection
        │
        ▼
Crossover
        │
        ▼
Mutation
        │
        ▼
New Population
```

---

## Simulated Annealing

Useful for exploring large solution spaces.

---

## Tabu Search

Can be used to escape local optimal solutions.

---

# Comparative Analysis

Compare:

```text
Exact Optimization
vs
Metaheuristic Optimization
```

Using:

* Objective value
* Solution quality
* Computational time
* Optimality gap
* Scalability

The results should answer:

> At what problem size does the exact optimization approach become computationally expensive?

And:

> Can heuristic methods produce sufficiently good solutions more efficiently?

---

# Uncertainty Analysis

The project should contain scenario-based experiments.

## Scenario A — Normal Operations

```text
Demand = Expected Demand
Resource Availability = 100%
```

---

## Scenario B — High Demand

```text
Demand = Expected Demand × 1.25
```

---

## Scenario C — Resource Disruption

```text
20% of Resources Unavailable
```

---

## Scenario D — Combined Disruption

```text
High Demand
+
Reduced Resource Availability
```

---

# Performance Metrics

The system should report:

## Operational Metrics

* Demand satisfaction rate
* Resource utilization
* Idle capacity
* Overtime
* Unmet demand

## Optimization Metrics

* Objective value
* Solver runtime
* Optimality gap
* Number of decision variables
* Number of constraints

---

# Visualization

Recommended visualizations include:

## Resource-Task Allocation Matrix

```text
           Task 1   Task 2   Task 3
Resource 1    ✓        -        ✓
Resource 2    -        ✓        -
Resource 3    ✓        ✓        -
```

---

## Gantt Chart

Visualize:

* Resource assignments
* Task duration
* Scheduling periods

---

## Demand vs Capacity

Compare:

```text
Demand
vs
Available Capacity
```

---

## Scenario Comparison

Compare:

```text
Cost
Resource Utilization
Unmet Demand
Overtime
```

across different uncertainty scenarios.

---

# Repository Structure

```text
resource-allocation-optimization/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   ├── processed/
│   │   └── README.md
│   │
│   └── scenarios/
│       ├── normal_demand.csv
│       ├── high_demand.csv
│       └── disruption_scenario.csv
│
├── notebooks/
│   ├── 01_problem_exploration.ipynb
│   ├── 02_deterministic_model.ipynb
│   ├── 03_multi_period_model.ipynb
│   ├── 04_uncertainty_analysis.ipynb
│   └── 05_metaheuristic_comparison.ipynb
│
├── src/
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── generator.py
│   │   └── validation.py
│   │
│   ├── models/
│   │   ├── deterministic_model.py
│   │   ├── multi_period_model.py
│   │   └── robust_model.py
│   │
│   ├── optimization/
│   │   ├── objective.py
│   │   ├── constraints.py
│   │   ├── solver.py
│   │   └── results.py
│   │
│   ├── heuristics/
│   │   ├── genetic_algorithm.py
│   │   ├── simulated_annealing.py
│   │   └── local_search.py
│   │
│   ├── scenarios/
│   │   ├── demand_uncertainty.py
│   │   └── resource_disruption.py
│   │
│   └── visualization/
│       ├── allocation_matrix.py
│       ├── gantt_chart.py
│       └── scenario_comparison.py
│
├── configs/
│   ├── model_config.yaml
│   ├── solver_config.yaml
│   └── experiment_config.yaml
│
├── experiments/
│   ├── small_scale/
│   ├── medium_scale/
│   ├── large_scale/
│   └── uncertainty/
│
├── results/
│   ├── optimization_results/
│   ├── heuristic_results/
│   ├── scenario_results/
│   └── figures/
│
├── docs/
│   ├── problem_definition.md
│   ├── mathematical_formulation.md
│   ├── assumptions.md
│   ├── computational_experiments.md
│   └── results_analysis.md
│
├── tests/
│   ├── test_data.py
│   ├── test_constraints.py
│   ├── test_objective.py
│   └── test_models.py
│
└── scripts/
    ├── run_deterministic_model.py
    ├── run_uncertainty_analysis.py
    ├── run_heuristics.py
    └── run_all_experiments.py
```

---

# Experimental Design

The computational experiments should progressively increase problem size.

## Small Scale

Example:

```text
10 Resources
5 Tasks
7 Time Periods
```

Purpose:

Validate the mathematical model.

---

## Medium Scale

Example:

```text
50 Resources
20 Tasks
30 Time Periods
```

Purpose:

Evaluate computational performance.

---

## Large Scale

Example:

```text
200+ Resources
50+ Tasks
Multiple Time Periods
```

Purpose:

Compare exact optimization and heuristics.

---

# Technology Stack

```text
Programming
└── Python

Mathematical Optimization
├── Pyomo
├── PuLP
└── OR-Tools

Solvers
├── HiGHS
├── CBC
├── GLPK
└── Gurobi (optional)

Data Processing
├── Pandas
├── NumPy
└── SciPy

Visualization
├── Matplotlib
└── Plotly

Development
├── Git
├── GitHub
└── Jupyter Notebook
```

---

# Implementation Roadmap

## Phase 1 — Problem Definition

* Select a specific resource allocation context.
* Define resources.
* Define tasks.
* Define planning periods.
* Identify costs and constraints.

---

## Phase 2 — Mathematical Formulation

Develop:

* Sets
* Parameters
* Decision variables
* Objective function
* Constraints

---

## Phase 3 — Deterministic Model

Implement and validate the baseline MILP model.

---

## Phase 4 — Multi-Period Extension

Add scheduling and temporal constraints.

---

## Phase 5 — Uncertainty

Introduce:

* Demand variability
* Resource disruptions

---

## Phase 6 — Metaheuristics

Implement at least one alternative solution approach.

---

## Phase 7 — Computational Analysis

Compare:

```text
Solution Quality
Runtime
Scalability
Robustness
```

---

# Expected Deliverables

The completed project should contain:

1. A complete mathematical formulation.
2. A deterministic MILP model.
3. A multi-period scheduling model.
4. Uncertainty scenarios.
5. At least one metaheuristic implementation.
6. Computational experiments.
7. Comparative performance analysis.
8. Decision-support visualizations.
9. A technical report.

---

# Example Research Questions

The final project should answer questions such as:

* How does demand uncertainty affect optimal resource allocation?
* What is the cost of resource disruptions?
* When does overtime become economically preferable to unmet demand?
* How does the exact optimization approach scale with problem size?
* How close are heuristic solutions to the optimal solution?

---

# Project Status

```text
Phase 1  → Problem Definition
Phase 2  → Mathematical Formulation
Phase 3  → Deterministic Optimization
Phase 4  → Multi-Period Scheduling
Phase 5  → Uncertainty Modelling
Phase 6  → Metaheuristic Optimization
Phase 7  → Computational Experiments
Phase 8  → Documentation and Publication
```

---

# Skills Demonstrated

This project demonstrates:

* Mathematical Modelling
* Operations Research
* Mixed-Integer Programming
* Resource Allocation
* Scheduling
* Optimization Algorithms
* Metaheuristics
* Uncertainty Modelling
* Computational Experimentation
* Python
* Decision Analytics
