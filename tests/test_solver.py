"""Unit tests for MILPSolver wrapper and HiGHS solver interface."""

import pytest
from src.optimization.solver import MILPSolver


def test_solver_initialization():
    """Tests that MILPSolver initializes properly with default configs."""
    solver = MILPSolver()
    assert solver.num_variables == 0
    assert solver.num_constraints == 0
    assert solver.status == "NOT_SOLVED"


def test_solver_simple_lp():
    """Tests solving a standard 2-variable LP to exact known optimum.
    min 2x + 3y
    s.t. x + y >= 4
         x >= 0, y >= 0
    Optimal solution: x=4, y=0, obj=8.0.
    """
    solver = MILPSolver()
    solver.add_variable("x", lb=0.0, obj_coeff=2.0)
    solver.add_variable("y", lb=0.0, obj_coeff=3.0)
    solver.add_constraint({"x": 1.0, "y": 1.0}, ">=", 4.0, name="c1")
    solver.set_objective_sense("minimize")

    res = solver.solve(log_to_console=False)
    assert res["status"] == "OPTIMAL"
    assert abs(res["objective_value"] - 8.0) < 1e-4
    assert abs(solver.get_var_value("x") - 4.0) < 1e-4
    assert abs(solver.get_var_value("y") - 0.0) < 1e-4


def test_solver_mip_integrality():
    """Tests solving a Mixed Integer Program with binary restrictions.
    min x + y
    s.t. 2x + 2y >= 3
         x, y in {0, 1}
    Optimal: one variable 1, the other 0, obj = 1.0.
    (If continuous, x=0.75, y=0.75 -> obj = 1.5, or x=1.5, y=0 -> obj = 1.5)
    """
    solver = MILPSolver()
    solver.add_variable("x", var_type="BINARY", obj_coeff=1.0)
    solver.add_variable("y", var_type="BINARY", obj_coeff=1.0)
    solver.add_constraint({"x": 2.0, "y": 2.0}, ">=", 3.0, name="knapsack")
    solver.set_objective_sense("minimize")

    res = solver.solve(log_to_console=False)
    assert res["status"] == "OPTIMAL"
    # To satisfy 2x + 2y >= 3 with binaries, both must be 1, obj = 2.0!
    assert abs(res["objective_value"] - 2.0) < 1e-4
    assert solver.get_var_value("x") == 1.0
    assert solver.get_var_value("y") == 1.0


def test_solver_infeasibility_detection():
    """Tests that infeasible models are correctly recognized."""
    solver = MILPSolver()
    solver.add_variable("x", lb=0.0, ub=2.0)
    solver.add_constraint({"x": 1.0}, ">=", 5.0)  # x in [0, 2] but x >= 5

    res = solver.solve(log_to_console=False)
    assert res["is_feasible"] is False
    assert res["status"] in ["INFEASIBLE", "PRIMAL_INFEASIBLE"]

