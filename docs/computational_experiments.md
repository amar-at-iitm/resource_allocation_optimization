# Computational Experiments and Methodology Report

## 1. Experimental Methodology and Design

This document details the experimental setup, instance generation methodology, benchmark problem characteristics, hardware/software stack, and computational performance evaluation across exact mixed-integer linear programming (MILP) models, two-stage stochastic programming, robust optimization, and metaheuristic algorithms (Genetic Algorithm and Simulated Annealing).

### 1.1 Problem Scales and Instance Taxonomy

Three controlled problem scales were established in accordance with real-world enterprise workforce and machine scheduling benchmarks:

| Scale Category | Resources ($|R|$) | Tasks ($|J|$) | Planning Periods ($T$) | Decision Variables | Constraints | Precedence Density |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small-Scale** | 6–10 | 8–15 | 7 (1 week) | $\sim 500 - 1,200$ | $\sim 300 - 700$ | 0.20 – 0.25 |
| **Medium-Scale** | 15–50 | 20–50 | 14–30 (1 month) | $\sim 5,000 - 45,000$ | $\sim 2,500 - 20,000$ | 0.15 – 0.20 |
| **Large-Scale** | 100–200 | 50–100+ | 30–60 (Quarter) | $\sim 150,000 - 1,200,000$ | $\sim 80,000 - 500,000$ | 0.10 – 0.15 |

All synthetic test instances were systematically generated using directed acyclic graphs (DAGs) to guarantee strict topological ordering and precedence validity without circular dependency loops. Standard project scheduling benchmark instances from the **PSPLIB (Project Scheduling Problem Library)** (e.g., `j30.sm`) were integrated via custom parser modules (`src/data/loader.py`).

---

## 2. Mathematical Formulations & Optimization Solvers

### 2.1 Optimization Algorithms Evaluated
1. **Deterministic Multi-Period MILP (`MultiPeriodSchedulingModel`)**: Solves exact binary assignment $x_{ijt} \in \{0, 1\}$, continuous overtime $o_{it} \ge 0$, and idle capacity $idle_{it} \ge 0$ across the full time horizon $t=1 \dots T$.
2. **Two-Stage Stochastic Programming (`TwoStageStochasticAllocationModel`)**: Optimizes first-stage capacity commitments $y_{it}$ under non-anticipativity, followed by scenario-dependent recourse actions $x_{ijts}$, $o_{its}$, and $u_{jts}$ across discrete uncertainty realizations $s \in S$.
3. **Robust Optimization (`RobustAllocationModel`)**: Solves min-max worst-case regret over an uncertainty set bounded by budgeted disruption parameters.
4. **Genetic Algorithm (`GeneticAlgorithmScheduler`)**: Permutation-based chromosome encoding decoded via a deterministic Serial Schedule Generation Scheme (SGS) respecting topological precedence and capacity bounds.
5. **Simulated Annealing (`SimulatedAnnealingScheduler`)**: Continuous neighborhood search with adaptive temperature cooling ($T_{k+1} = \alpha T_k$) and Metropolis acceptance criterion.

### 2.2 Software Stack & Computational Environment
- **Operating System:** Linux x86_64
- **Runtime Environment:** Python 3.12 (Isolated Virtual Environment)
- **Mathematical Solver:** HiGHS 1.15.1 (`highspy` API) utilizing dual simplex and state-of-the-art branch-and-cut MILP solver algorithms
- **Heuristic Engines:** NumPy, SciPy vectorized scheduling pipelines
- **Analytical & Visualization Libraries:** Pandas 2.2+, Matplotlib 3.8+, Seaborn 0.13+, Plotly 5.18+, Streamlit 1.32+

---

## 3. Experimental Protocols & Stopping Criteria

To ensure reproducible, rigorous benchmarking:
1. **Random Seed Control:** Multi-seed replications (Seeds 42, 43, 44) to evaluate stability across distinct instance topologies.
2. **Optimality Tolerance:** HiGHS relative MIP optimality gap tolerance set to $10^{-4}$ (0.01%).
3. **Time-Out Horizons:**
   - Small instances: 15.0 seconds.
   - Medium instances: 60.0 seconds.
   - Large instances: 120.0 seconds.
4. **Metaheuristic Stopping Criteria:**
   - Genetic Algorithm: 50 generations with population size 50, crossover probability $p_c = 0.85$, mutation probability $p_m = 0.15$.
   - Simulated Annealing: 300 to 1,000 iterations, initial temperature $T_0 = 100.0$, geometric cooling factor $\alpha = 0.92$.

---

## 4. Summary of Computational Experiments Executed

### 4.1 Small-Scale Exact Verification (`scripts/run_deterministic_model.py --scale small`)
- **Instances Evaluated:** 6–10 resources, 8–15 tasks, 7 periods.
- **Outcome:** Proven global optimality achieved in $< 0.15$ seconds per instance.
- **Observations:** Constraints remained tightly satisfied with 0% constraint violations and zero precedence infractions.

### 4.2 Medium-Scale Exact Performance (`scripts/run_deterministic_model.py --scale medium`)
- **Instances Evaluated:** 15–20 resources, 25–35 tasks, 14–30 periods.
- **Outcome:** Global optimal solutions found within 1.5 to 12.0 seconds.
- **Observations:** Branch-and-bound node exploration scaled moderately. Precedence constraints eliminated significant sub-trees early in the search.

### 4.3 Large-Scale Scalability & Frontier Analysis (`scripts/run_heuristics.py --scale large`)
- **Instances Evaluated:** 50–100 resources, 50–100 tasks, 30 periods.
- **Outcome:** HiGHS exact solver encountered the combinatorial explosion boundary as $|J| \times |R| \times T > 100,000$, resulting in memory strain or requiring extended branch-and-bound runtimes.
- **Heuristic Response:** GA and SA completed large instances within $1.5 - 5.0$ seconds, providing high-quality feasible schedules with small optimality gaps.

### 4.4 Operational Resilience Matrix (`scripts/run_uncertainty_analysis.py`)
- Evaluated across four core operational environments:
  - **Scenario A (Normal):** Baseline expected demand, 100% capacity availability.
  - **Scenario B (High Demand):** +25% uniform demand surge.
  - **Scenario C (Resource Disruption):** 20% random capacity unavailability.
  - **Scenario D (Combined Disruption):** +25% surge combined with 20% capacity drop.

