# Model Assumptions & Methodological Scope

## 1. Operational Assumptions

1. **Discrete Time Horizon:** Time is discretized into uniform periods (e.g., shifts, days, or weeks) $t \in \{0, 1, \dots, T-1\}$. Activities and resource capacities are calibrated to period granularity.
2. **Non-Preemptive Execution:** Once a task starts execution, it runs uninterrupted for $d_j$ consecutive periods unless an explicit multi-mode or preemption setting is active.
3. **Resource Non-Splitting:** An individual task is performed by one dedicated qualified resource at any given period rather than split across multiple resources unless pooled.
4. **Deterministic Base Case:** Baseline parameters (capacities, standard requirements, baseline demands) are treated as known with certainty; stochastic shocks are introduced systematically through explicit scenarios.
5. **Overtime Economics:** Overtime capacity is strictly bounded ($O_{it}^{\max}$) and penalized at a rate $c_{it}^{\text{over}} > c_{it}^{\text{reg}}$, ensuring overtime is only scheduled when economically preferable to unmet demand penalties.
6. **Acyclic Precedence:** Precedence constraints form a valid Directed Acyclic Graph (DAG) with no circular dependencies.

---

## 2. Linearization and Computational Properties

- **Linearity:** The objective function and all constraint equations are linear with respect to decision variables ($x_{ijt}, s_{jt}, y_j, o_{it}, idle_{it}, u_{jt}$), ensuring the problem is solvable by branch-and-cut / simplex MILP solvers (HiGHS).
- **NP-Hardness:** Because the problem subsumes the classic Resource-Constrained Project Scheduling Problem (RCPSP) and the Generalized Assignment Problem (GAP), the multi-period formulation is $\mathcal{NP}$-hard in the strong sense.
- **Tractability Boundary:** Exact optimization is effective for small to medium scales (up to ~50 resources, ~40 tasks). For larger problem dimensions, exact solvers provide upper/lower bounds while metaheuristics (Genetic Algorithm, Simulated Annealing) deliver rapid near-optimal solutions.

