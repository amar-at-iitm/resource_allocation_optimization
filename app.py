"""Interactive Streamlit Decision-Support Dashboard for Multi-Period Resource Allocation and Scheduling."""

import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.generator import generate_instance
from src.data.schemas import ProblemInstance, AllocationSolution
from src.scenarios.demand_uncertainty import DemandScenarioGenerator
from src.scenarios.resource_disruption import ResourceDisruptionGenerator
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.heuristics.genetic_algorithm import GeneticAlgorithm
from src.heuristics.simulated_annealing import SimulatedAnnealing
from src.visualization.gantt_chart import plot_gantt
from src.visualization.allocation_matrix import plot_allocation_matrix
from src.visualization.scenario_comparison import (
    plot_demand_vs_capacity,
    plot_scenario_comparison,
    plot_tradeoff_curve,
)

st.set_page_config(
    page_title="Multi-Period Resource Allocation Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Header
st.title("⚡ Multi-Period Resource Allocation & Scheduling Platform")
st.markdown(
    "Decision-support system for constrained multi-period scheduling under demand uncertainty and supply disruption. "
    "Compare **Exact MILP (HiGHS)** against **Metaheuristic Algorithms (GA / SA)** and evaluate resilience strategies."
)

# Sidebar Configuration
st.sidebar.header("⚙️ Problem & Solver Controls")

scale_choice = st.sidebar.selectbox(
    "Problem Scale",
    options=["Small (6 Res, 8 Tasks, 7 Periods)", "Medium (15 Res, 25 Tasks, 14 Periods)", "Large (30 Res, 50 Tasks, 21 Periods)"],
    index=0
)

solver_choice = st.sidebar.selectbox(
    "Optimization Method",
    options=["Exact MILP (HiGHS)", "Genetic Algorithm (GA)", "Simulated Annealing (SA)"],
    index=0
)

scenario_choice = st.sidebar.selectbox(
    "Uncertainty Scenario",
    options=[
        "Normal (Baseline Demand & 100% Capacity)",
        "High Demand (+25% Demand Surge)",
        "Resource Disruption (20% Capacity Loss)",
        "Combined Stress (Surge + Disruption)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Parameter Overrides")
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
ot_weight = st.sidebar.slider("Overtime Cost Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
unmet_weight = st.sidebar.slider("Unmet Demand Penalty ($/hr)", min_value=100.0, max_value=2000.0, value=500.0, step=50.0)

run_button = st.sidebar.button("🚀 Run Optimization & Analysis", type="primary", use_container_width=True)

# Helper function to generate and configure instance
def setup_instance(scale_label: str, scenario_label: str, rand_seed: int) -> ProblemInstance:
    if "Small" in scale_label:
        inst = generate_instance(n_resources=6, n_tasks=8, n_periods=7, seed=rand_seed, precedence_density=0.25)
    elif "Medium" in scale_label:
        inst = generate_instance(n_resources=15, n_tasks=25, n_periods=14, seed=rand_seed, precedence_density=0.15)
    else:
        inst = generate_instance(n_resources=30, n_tasks=50, n_periods=21, seed=rand_seed, precedence_density=0.10)

    # Adjust cost parameters
    for r in inst.resources.values():
        r.overtime_cost_per_period = r.regular_cost_per_period * ot_weight
    for t in inst.tasks.values():
        t.penalty_cost = unmet_weight

    # Apply Scenario
    if "High Demand" in scenario_label:
        dgen = DemandScenarioGenerator(seed=rand_seed)
        inst = dgen.apply_scenario_to_instance(inst, scenario_type="high_demand", multiplier=1.25)
    elif "Resource Disruption" in scenario_label:
        rgen = ResourceDisruptionGenerator(seed=rand_seed)
        inst = rgen.apply_disruption_to_instance(inst, scenario_type="capacity_reduction", reduction_factor=0.20)
    elif "Combined" in scenario_label:
        dgen = DemandScenarioGenerator(seed=rand_seed)
        rgen = ResourceDisruptionGenerator(seed=rand_seed)
        inst = dgen.apply_scenario_to_instance(inst, scenario_type="high_demand", multiplier=1.25)
        inst = rgen.apply_disruption_to_instance(inst, scenario_type="capacity_reduction", reduction_factor=0.20)

    return inst

# Execution State Management
if "last_solution" not in st.session_state or run_button:
    with st.spinner("Executing optimization model..."):
        instance = setup_instance(scale_choice, scenario_choice, int(seed))
        start_t = time.perf_counter()

        if solver_choice == "Exact MILP (HiGHS)":
            model = MultiPeriodSchedulingModel(instance)
            solution = model.solve(time_limit=60.0)
        elif solver_choice == "Genetic Algorithm (GA)":
            ga = GeneticAlgorithm(instance, seed=int(seed))
            solution = ga.solve(generations=40, pop_size=30)
        else:
            sa = SimulatedAnnealing(instance, seed=int(seed))
            solution = sa.solve(initial_temp=100.0, cooling_rate=0.92, max_iter=300)

        elapsed = time.perf_counter() - start_t
        solution.solve_time_seconds = elapsed

        st.session_state["instance"] = instance
        st.session_state["solution"] = solution
        st.session_state["solver_name"] = solver_choice

# Display KPIs
instance = st.session_state.get("instance")
solution = st.session_state.get("solution")
solver_name = st.session_state.get("solver_name")

if instance and solution:
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Operational Cost", f"${solution.objective_cost:,.2f}")
    with col2:
        st.metric("Solve Status", f"{solution.status}")
    with col3:
        st.metric("Demand Satisfaction", f"{solution.demand_satisfaction_percent:.1f}%")
    with col4:
        st.metric("Total Overtime Hours", f"{solution.total_overtime_hours:.1f} hrs")
    with col5:
        st.metric("Execution Time", f"{solution.solve_time_seconds:.3f} s")

    st.markdown("---")

    # Tabs for Visualizations
    tab_gantt, tab_matrix, tab_demand_cap, tab_scenarios, tab_data = st.tabs([
        "📊 Interactive Gantt Schedule",
        "🗓️ Workload Allocation Matrix",
        "📈 Demand vs. Capacity Buffers",
        "🛡️ Uncertainty & Scenario Resilience",
        "📋 Solution Raw Data"
    ])

    with tab_gantt:
        st.subheader("Resource Schedule Timeline")
        st.markdown("Interactive Gantt diagram displaying assigned tasks, durations, and skill requirements per resource.")
        fig_gantt = plot_gantt(
            solution,
            instance,
            title=f"Schedule Gantt Chart — {solver_name}"
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

    with tab_matrix:
        st.subheader("Resource-Period Workload Heatmap")
        st.markdown("Workload intensity across time periods highlighting resource utilization peaks.")
        fig_matrix = plot_allocation_matrix(
            solution,
            instance,
            title=f"Workload Heatmap (Hours) — {solver_name}"
        )
        st.pyplot(fig_matrix)

    with tab_demand_cap:
        st.subheader("Planning Horizon Demand vs. Capacity")
        st.markdown("Comparison between regular nominal capacity, overtime buffer capacity, and scheduled workload.")
        fig_dvc = plot_demand_vs_capacity(
            instance,
            solution,
            title="Aggregate Period Demand vs. Available Resource Capacity"
        )
        st.pyplot(fig_dvc)

    with tab_scenarios:
        st.subheader("Multi-Scenario Resilience Benchmarks")
        scenario_csv = ROOT_DIR / "results" / "scenario_results" / "uncertainty_matrix_results.csv"
        if scenario_csv.exists():
            df_scenarios = pd.read_csv(scenario_csv)
            st.dataframe(df_scenarios, use_container_width=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_sc = plot_scenario_comparison(df=df_scenarios)
                st.pyplot(fig_sc)
            with col_s2:
                fig_to = plot_tradeoff_curve(df=df_scenarios)
                st.pyplot(fig_to)
        else:
            st.info("Run `scripts/run_uncertainty_analysis.py` to populate multi-scenario benchmarks.")

    with tab_data:
        st.subheader("Detailed Assignment & KPI Data")
        if solution.assignments:
            assign_records = [
                {
                    "Resource": a.resource_id,
                    "Task": a.task_id,
                    "Period": a.period,
                    "Assigned Work (hrs)": a.assigned_work
                }
                for a in solution.assignments
            ]
            st.dataframe(pd.DataFrame(assign_records), use_container_width=True)
        else:
            st.write("No granular assignments recorded for this configuration.")
