# Mathematical Formulation: Multi-Period Resource Allocation and Scheduling under Uncertainty

## 1. Problem Overview

The Multi-Period Resource Allocation and Scheduling problem involves optimally assigning a heterogeneous pool of constrained resources to a set of competing tasks over a discrete planning horizon $T$. 

The model guarantees:
1. Operational skill compatibility.
2. Resource capacity and working hour limits.
3. Overtime thresholds and costs.
4. Precedence relationships among tasks (DAG).
5. Multi-period continuity and demand satisfaction.
6. Minimization of total operational cost, idle costs, and unmet demand penalties.

---

## 2. Sets and Indices

| Symbol | Definition |
| :--- | :--- |
| $I$ | Set of resources, indexed by $i \in \{1, 2, \dots, \|I\|\}$ |
| $J$ | Set of tasks (activities), indexed by $j \in \{1, 2, \dots, \|J\|\}$ |
| $T$ | Set of discrete time periods, indexed by $t \in \{0, 1, \dots, \|T\|-1\}$ |
| $S$ | Set of required skill types, indexed by $s \in S$ |
| $P_j$ | Set of immediate predecessors of task $j$, $P_j \subset J$ |
| $\Omega$ | Set of operational scenarios (for stochastic extensions), indexed by $\omega \in \Omega$ |

---

## 3. Parameters

### 3.1 Resource Parameters
- $C_{it} \ge 0$: Regular capacity of resource $i$ in period $t$ (e.g., standard hours, typically 8.0).
- $O_{it}^{\max} \ge 0$: Maximum permissible overtime for resource $i$ in period $t$.
- $c_{it}^{\text{reg}} \ge 0$: Cost per unit of regular capacity utilized by resource $i$ in period $t$.
- $c_{it}^{\text{over}} > c_{it}^{\text{reg}}$: Cost per unit of overtime utilized by resource $i$ in period $t$.
- $c_{it}^{\text{idle}} \ge 0$: Opportunity or holding cost per unit of unused regular capacity.
- $qual_{is} \in \{0, 1\}$: 1 if resource $i$ possesses skill $s$, 0 otherwise.

### 3.2 Task Parameters
- $d_j \in \mathbb{Z}_{\ge 1}$: Duration of task $j$ (in planning periods).
- $r_{j} > 0$: Resource demand (units of capacity per period) required to execute task $j$.
- $req_{js} \in \{0, 1\}$: 1 if task $j$ requires skill $s$, 0 otherwise.
- $p_j \ge 0$: Penalty per unit of unmet or delayed demand for task $j$.
- $ES_j \ge 0$: Earliest possible start period (release date) for task $j$.
- $LF_j \le \|T\|$: Latest permissible finish period (due date) for task $j$.

---

## 4. Decision Variables

### 4.1 Primary Assignment and Scheduling Variables
- $x_{ijt} \in \{0, 1\}$: 1 if resource $i$ is assigned to execute task $j$ during period $t$; 0 otherwise.
- $s_{jt} \in \{0, 1\}$: 1 if task $j$ starts at the beginning of period $t$; 0 otherwise.
- $y_j \in \{0, 1\}$: 1 if task $j$ is completed within the planning horizon; 0 if cancelled / unmet.

### 4.2 Resource Utilization Variables
- $o_{it} \ge 0$: Overtime hours worked by resource $i$ in period $t$.
- $idle_{it} \ge 0$: Unused regular capacity of resource $i$ in period $t$.
- $u_{jt} \ge 0$: Unmet demand (shortfall) for task $j$ in period $t$.

---

## 5. Objective Function

The objective is to minimize total operational expenditure:

$$\min Z = \sum_{i \in I} \sum_{t \in T} \left( c_{it}^{\text{reg}} \sum_{j \in J} r_j x_{ijt} + c_{it}^{\text{over}} o_{it} + c_{it}^{\text{idle}} idle_{it} \right) + \sum_{j \in J} \sum_{t \in T} p_j u_{jt}$$

Alternatively, when regular labor cost is contracted as a fixed baseline salary:
$$\min Z' = \sum_{i \in I} \sum_{t \in T} \left( c_{it}^{\text{over}} o_{it} + c_{it}^{\text{idle}} idle_{it} \right) + \sum_{j \in J} p_j (1 - y_j)$$

---

## 6. Constraints

### 6.1 Capacity and Overtime Balance
For each resource $i \in I$ and each time period $t \in T$:
$$\sum_{j \in J} r_j \cdot x_{ijt} + idle_{it} = C_{it} + o_{it}$$

### 6.2 Overtime Upper Bound
For each resource $i \in I$ and period $t \in T$:
$$0 \le o_{it} \le O_{it}^{\max}$$

### 6.3 Skill Qualification Constraint
A resource cannot be assigned to a task if lacking the requisite skill:
$$x_{ijt} \le \sum_{s \in S} qual_{is} \cdot req_{js}, \quad \forall i \in I, j \in J, t \in T$$

### 6.4 Single Resource Allocation per Task
For non-preemptive tasks, exactly one qualified resource is assigned to task $j$ throughout its active duration:
$$\sum_{i \in I} x_{ijt} \le 1, \quad \forall j \in J, t \in T$$

### 6.5 Task Start and Duration Continuity
Each task $j$ starts at most once:
$$\sum_{t = ES_j}^{LF_j - d_j} s_{jt} = y_j, \quad \forall j \in J$$

If task $j$ starts at period $t$, it must remain active for $d_j$ consecutive periods:
$$\sum_{i \in I} x_{ij, t + \tau} \ge s_{jt}, \quad \forall j \in J, t \in \{ES_j, \dots, LF_j - d_j\}, \tau \in \{0, \dots, d_j - 1\}$$

Total active periods equals required duration if fulfilled:
$$\sum_{t \in T} \sum_{i \in I} x_{ijt} = d_j \cdot y_j, \quad \forall j \in J$$

### 6.6 Precedence Constraints
If task $k \in P_j$ precedes task $j$, task $j$ cannot start until task $k$ has finished:
$$\sum_{t \in T} t \cdot s_{jt} \ge \sum_{t \in T} (t + d_k) \cdot s_{kt}, \quad \forall j \in J, k \in P_j$$

### 6.7 Non-Negativity and Integrality
$$x_{ijt} \in \{0, 1\}, \quad s_{jt} \in \{0, 1\}, \quad y_j \in \{0, 1\}$$
$$o_{it} \ge 0, \quad idle_{it} \ge 0, \quad u_{jt} \ge 0$$

