## Dataset and Benchmark Instances

### Overview

Unlike conventional machine-learning projects, Operations Research projects do not always depend on a large observational dataset.

This project focuses on a mathematical optimization problem involving:

- Limited resources
- Multiple tasks
- Resource capacities
- Scheduling periods
- Task requirements
- Operational constraints
- Uncertainty

Therefore, the project will use a combination of:

```text
Established Operations Research Benchmarks
        +
Synthetic Uncertainty Scenarios
        +
Custom Instance Generator
```

The benchmark instances provide standardized optimization problems, while the synthetic scenario generator allows controlled experiments involving uncertainty and resource disruptions.

---

# Primary Benchmark Source

The primary benchmark source is the **Operations Research and Scheduling Research Group Project Database** maintained by Ghent University.

The database contains benchmark datasets for several scheduling and resource-constrained project scheduling problems, including resource-constrained project scheduling, multi-skilled scheduling, and other related optimization problems. The database provides thousands of benchmark instances and, for some problem classes, substantially larger collections.

---

## Benchmark Dataset Download

**Operations Research and Scheduling Research Group Project Database:**

[OR&S Project Data Repository](https://www.projectmanagement.ugent.be/research/data?utm_source=chatgpt.com)

The repository provides downloadable benchmark instances for multiple resource-constrained scheduling problems.

---

# Recommended Benchmark: Resource-Constrained Project Scheduling Problem

The initial version of this project will use instances based on the:

## Resource-Constrained Project Scheduling Problem (RCPSP)

The RCPSP involves scheduling a set of activities subject to:

- Precedence constraints
- Limited resource availability
- Activity durations
- Resource requirements

The objective is commonly to minimize the project completion time.

The problem structure naturally extends to multi-period resource allocation and scheduling.

---

# Core Benchmark Instance Types

The benchmark repository includes several RCPSP datasets.

For this project, instances can be selected progressively based on problem size.

### Small-Scale Instances

Suitable for:

- Model validation
- Debugging
- Mathematical verification
- Comparison with known solutions

---

### Medium-Scale Instances

Suitable for:

- Computational experiments
- Solver comparisons
- Constraint sensitivity analysis

---

### Large-Scale Instances

Suitable for:

- Scalability analysis
- Metaheuristic experiments
- Exact-versus-heuristic comparison

The benchmark database contains collections ranging from relatively small instances to substantially larger datasets.

---

# Recommended Instance Selection Strategy

The project should use three experimental groups.

## Group A — Small Instances

Purpose:

```text
Validate Mathematical Model
+
Verify Solver Results
```

Recommended characteristics:

```text
Approximately 10–30 Activities
Few Resource Types
Short Planning Horizon
```

---

## Group B — Medium Instances

Purpose:

```text
Evaluate Computational Performance
```

Recommended characteristics:

```text
Approximately 30–100 Activities
Multiple Resource Types
More Complex Precedence Structures
```

---

## Group C — Large Instances

Purpose:

```text
Compare Exact Optimization
vs
Metaheuristic Approaches
```

Recommended characteristics:

```text
100+ Activities
Multiple Limited Resources
Complex Scheduling Structure
```

The exact selection should depend on the computational capabilities and optimization solver used.

---

# Benchmark Data Structure

Each scheduling instance conceptually contains information about:

## Activities

Each activity may have:

```text
Activity ID
Duration
Predecessor Activities
```

---

## Resources

Resources may have:

```text
Resource Type
Available Capacity
Availability Over Time
```

---

## Activity Resource Requirements

Each activity requires specific quantities of resources.

Conceptually:

```text
Activity A
    │
    ├── Resource 1 → 2 Units
    ├── Resource 2 → 1 Unit
    └── Resource 3 → 3 Units
```

---

# Mapping Benchmark Data to This Project

The benchmark data will be transformed into the project's general optimization framework.

```text
Benchmark Component
        ↓
Project Component

Activity
        ↓
Task

Resource Type
        ↓
Resource Category

Resource Capacity
        ↓
Available Capacity

Activity Duration
        ↓
Task Duration

Precedence Constraint
        ↓
Scheduling Constraint
```

This mapping allows the project to maintain a general resource-allocation interpretation while using established benchmark instances.

---

# Custom Data Generator

A custom synthetic data generator will also be developed.

The generator will create additional optimization instances with configurable:

- Number of resources
- Number of tasks
- Number of planning periods
- Resource capacities
- Task durations
- Resource costs
- Task priorities
- Demand variability
- Resource disruption probability

Example configuration:

```python
config = {
    "n_resources": 50,
    "n_tasks": 100,
    "n_periods": 30,
    "demand_variability": 0.20,
    "resource_disruption_probability": 0.10,
    "random_seed": 42
}
```

---

# Synthetic Uncertainty Scenarios

The benchmark datasets generally represent deterministic optimization problems.

To investigate uncertainty, the project will generate controlled scenarios.

---

## Scenario 1 — Baseline

```text
Expected Demand
+
Full Resource Availability
```

This scenario represents normal operations.

---

## Scenario 2 — Increased Demand

Resource or task requirements are increased.

Example:

```text
Demand Increase = 20%
```

The objective is to evaluate how the optimal allocation changes.

---

## Scenario 3 — Resource Disruption

A portion of the available resources becomes temporarily unavailable.

Example:

```text
Resource Availability = 80%
```

This may represent:

- Employee absenteeism
- Machine failure
- Equipment downtime

---

## Scenario 4 — Combined Uncertainty

```text
Higher Demand
+
Reduced Resource Availability
```

This represents a stressed operational environment.

---

# Synthetic Parameter Generation

Additional parameters required for the multi-period resource allocation model may be generated using documented distributions.

## Resource Costs

For example:

```text
Regular Cost:
Random Distribution Based on Resource Type

Overtime Cost:
Regular Cost × Overtime Multiplier
```

---

## Resource Availability

Availability can vary by:

- Time period
- Resource
- Scenario

---

## Task Priorities

Tasks can be assigned different penalty values.

Example:

```text
High Priority
    ↓
High Unmet-Demand Penalty

Medium Priority
    ↓
Moderate Penalty

Low Priority
    ↓
Lower Penalty
```

---

# Reproducibility

All synthetic instances must be generated using controlled random seeds.

Example:

```python
RANDOM_SEED = 42
```

Each experiment should store:

```text
Experiment Configuration
Random Seed
Instance Identifier
Solver Configuration
Runtime
Objective Value
Optimality Gap
```

This ensures that computational experiments can be reproduced.

---

# Dataset Storage Structure

The project repository will organize benchmark and generated data as follows:

```text
data/
│
├── benchmarks/
│   ├── small/
│   ├── medium/
│   └── large/
│
├── generated/
│   ├── deterministic/
│   └── stochastic/
│
└── scenarios/
    ├── baseline/
    ├── high_demand/
    ├── resource_disruption/
    └── combined_disruption/
```

---

# Data Preprocessing

Benchmark instances will be parsed into a standardized internal format.

Example:

```text
task_id
duration
predecessors
resource_requirements
priority
```

Resource information may be represented as:

```text
resource_id
resource_type
capacity
availability
cost
```

This standardized representation allows different benchmark sources and generated instances to be processed by the same optimization framework.

---

# Why Benchmark Instances Are Used

The use of established benchmark datasets provides several advantages.

### 1. Model Validation

The optimization model can be tested against known benchmark instances.

### 2. Reproducibility

Other researchers and practitioners can reproduce the computational experiments.

### 3. Fair Algorithm Comparison

Exact optimization and metaheuristic algorithms can be evaluated on identical problem instances.

### 4. Scalability Analysis

The project can systematically investigate how computational performance changes as problem size increases.

---

# Dataset Limitations

Benchmark instances may not contain:

- Real employee names
- Real organizational costs
- Actual workforce attendance
- Commercially observed demand

However, this is not necessarily a weakness.

The primary purpose of the project is to evaluate:

```text
Mathematical Formulation
+
Optimization Algorithms
+
Computational Performance
+
Robustness Under Uncertainty
```

The benchmark instances provide controlled and standardized experimental conditions for this purpose.

---

# Final Data Architecture

```text
OR Benchmark Instances
        │
        ▼
Standardized Problem Format
        │
        ├────────────────┐
        │                │
        ▼                ▼
Deterministic Model   Scenario Generator
        │                │
        │                ├── High Demand
        │                ├── Resource Failure
        │                └── Combined Disruption
        │
        └────────┬───────┘
                 ▼
      Optimization Framework
                 │
        ┌────────┴─────────┐
        ▼                  ▼
Exact Methods         Metaheuristics
        │                  │
        └────────┬─────────┘
                 ▼
     Computational Comparison
```

---

# Recommended Data Usage Policy

The repository should avoid committing large benchmark archives unless redistribution is explicitly permitted.

Instead, the repository should provide:

1. Dataset download instructions.
2. A benchmark parser.
3. A standardized preprocessing pipeline.
4. Scripts for generating synthetic scenarios.
5. Experiment configurations.

Users should download benchmark datasets from the original source and place them in:

```text
data/benchmarks/raw/
```

The preprocessing scripts will convert them into the standardized project format.

---

# Future Dataset Extensions

The framework can later be extended using benchmark datasets for:

- Multi-skilled resource-constrained project scheduling
- Multi-mode project scheduling
- Workforce scheduling
- Nurse rostering
- Machine scheduling
- Vehicle scheduling

This will allow the optimization framework to evolve into a more general Operations Research experimentation platform.