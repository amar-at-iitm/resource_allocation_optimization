"""High-level MILP Solver Abstraction Layer using HiGHS (highspy)."""

from typing import Any, Dict, List, Optional, Tuple, Union
import highspy

from src.data.config_loader import SolverConfig, load_solver_config


class MILPSolver:
    """Object-oriented wrapper around highspy for Mixed-Integer Linear Programming."""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or load_solver_config()
        self.highs = highspy.Highs()
        self._var_name_to_idx: Dict[str, int] = {}
        self._var_idx_to_name: Dict[int, str] = {}
        self._var_types: Dict[int, str] = {}
        self._constraint_names: List[str] = []
        self._is_solved: bool = False
        self._status: str = "NOT_SOLVED"
        self._objective_value: Optional[float] = None
        self._solution_values: Dict[str, float] = {}
        self._solve_time: float = 0.0

        # Apply default solver configurations
        self._configure_solver()

    def _configure_solver(self) -> None:
        """Applies configured parameters to the HiGHS instance."""
        self.highs.setOptionValue("time_limit", float(self.config.time_limit_seconds))
        self.highs.setOptionValue("mip_rel_gap", float(self.config.mip_rel_gap))
        self.highs.setOptionValue("mip_abs_gap", float(self.config.mip_abs_gap))
        self.highs.setOptionValue("output_flag", bool(self.config.log_to_console))
        if self.config.parallel_threads > 1:
            self.highs.setOptionValue("threads", int(self.config.parallel_threads))
        if self.config.presolve:
            self.highs.setOptionValue("presolve", str(self.config.presolve))

    def add_variable(
        self,
        name: str,
        lb: float = 0.0,
        ub: float = float("inf"),
        var_type: str = "CONTINUOUS",
        obj_coeff: float = 0.0
    ) -> int:
        """Adds a decision variable to the optimization model.
        
        Args:
            name: Unique name for the variable.
            lb: Lower bound.
            ub: Upper bound.
            var_type: 'CONTINUOUS', 'INTEGER', or 'BINARY'.
            obj_coeff: Linear objective coefficient.
            
        Returns:
            Column index of the variable.
        """
        if name in self._var_name_to_idx:
            raise ValueError(f"Variable '{name}' already exists in solver.")

        var_type_upper = var_type.upper()
        if var_type_upper == "BINARY":
            lb = max(0.0, lb)
            ub = min(1.0, ub)
            highs_type = highspy.HighsVarType.kInteger
        elif var_type_upper == "INTEGER":
            highs_type = highspy.HighsVarType.kInteger
        else:
            highs_type = highspy.HighsVarType.kContinuous

        ub_val = highspy.kHighsInf if (ub == float("inf") or ub >= 1e20) else float(ub)
        lb_val = -highspy.kHighsInf if (lb == float("-inf") or lb <= -1e20) else float(lb)

        idx = self.highs.getNumCol()
        self.highs.addVar(lb_val, ub_val)
        self.highs.changeColIntegrality(idx, highs_type)
        if obj_coeff != 0.0:
            self.highs.changeColCost(idx, float(obj_coeff))

        self._var_name_to_idx[name] = idx
        self._var_idx_to_name[idx] = name
        self._var_types[idx] = var_type_upper
        return idx

    def add_constraint(
        self,
        terms: Dict[Union[str, int], float],
        sense: str,
        rhs: float,
        name: str = ""
    ) -> int:
        """Adds a linear constraint of the form sum(coeff * var) [<=, >=, ==] rhs.
        
        Args:
            terms: Mapping from variable name or index to coefficient.
            sense: '<=', '>=', or '==' ('=').
            rhs: Right-hand side scalar constant.
            name: Optional descriptive name for the constraint.
            
        Returns:
            Row index of the constraint.
        """
        indices: List[int] = []
        coeffs: List[float] = []

        for var_key, coeff in terms.items():
            if coeff == 0.0:
                continue
            if isinstance(var_key, str):
                if var_key not in self._var_name_to_idx:
                    raise KeyError(f"Variable '{var_key}' not found in model.")
                v_idx = self._var_name_to_idx[var_key]
            else:
                v_idx = var_key

            indices.append(v_idx)
            coeffs.append(float(coeff))

        num_nz = len(indices)
        rhs_val = float(rhs)

        if sense in ["<=", "<"]:
            row_lb = -highspy.kHighsInf
            row_ub = rhs_val
        elif sense in [">=", ">"]:
            row_lb = rhs_val
            row_ub = highspy.kHighsInf
        elif sense in ["==", "="]:
            row_lb = rhs_val
            row_ub = rhs_val
        else:
            raise ValueError(f"Unknown constraint sense: '{sense}'. Use '<=', '>=', or '=='.")

        c_idx = self.highs.getNumRow()
        self.highs.addRow(row_lb, row_ub, num_nz, indices, coeffs)
        self._constraint_names.append(name or f"c_{c_idx}")
        return c_idx

    def set_objective_sense(self, sense: str = "minimize") -> None:
        """Sets optimization sense ('minimize' or 'maximize')."""
        if sense.lower().startswith("min"):
            self.highs.changeObjectiveSense(highspy.ObjSense.kMinimize)
        elif sense.lower().startswith("max"):
            self.highs.changeObjectiveSense(highspy.ObjSense.kMaximize)
        else:
            raise ValueError(f"Unknown objective sense: '{sense}'")

    def solve(
        self,
        time_limit: Optional[float] = None,
        mip_rel_gap: Optional[float] = None,
        log_to_console: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Executes the solver and extracts optimal solution."""
        if time_limit is not None:
            self.highs.setOptionValue("time_limit", float(time_limit))
        if mip_rel_gap is not None:
            self.highs.setOptionValue("mip_rel_gap", float(mip_rel_gap))
        if log_to_console is not None:
            self.highs.setOptionValue("output_flag", bool(log_to_console))

        self.highs.run()
        self._solve_time = float(self.highs.getRunTime())
        model_status = self.highs.getModelStatus()

        # Map HiGHS status enum to standardized string
        status_map = {
            highspy.HighsModelStatus.kOptimal: "OPTIMAL",
            highspy.HighsModelStatus.kInfeasible: "INFEASIBLE",
            highspy.HighsModelStatus.kUnbounded: "UNBOUNDED",
            highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
            highspy.HighsModelStatus.kIterationLimit: "ITERATION_LIMIT",
        }
        self._status = status_map.get(model_status, str(model_status).replace("HighsModelStatus.k", "").upper())

        is_feasible = self._status in ["OPTIMAL", "TIME_LIMIT"]

        if is_feasible:
            self._objective_value = float(self.highs.getObjectiveValue())
            col_values = self.highs.getSolution().col_value
            self._solution_values = {
                name: float(col_values[idx])
                for name, idx in self._var_name_to_idx.items()
            }
        else:
            self._objective_value = None
            self._solution_values = {}

        self._is_solved = True

        return {
            "status": self._status,
            "is_feasible": is_feasible,
            "objective_value": self._objective_value,
            "solve_time_seconds": self._solve_time,
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints
        }

    @property
    def status(self) -> str:
        return self._status

    @property
    def objective_value(self) -> Optional[float]:
        return self._objective_value

    @property
    def solve_time(self) -> float:
        return self._solve_time

    @property
    def num_variables(self) -> int:
        return self.highs.getNumCol()

    @property
    def num_constraints(self) -> int:
        return self.highs.getNumRow()

    def get_var_value(self, name: str) -> float:
        """Returns the solution value for a variable by name."""
        if not self._is_solved:
            raise RuntimeError("Model has not been solved yet.")
        if name not in self._solution_values:
            raise KeyError(f"Variable '{name}' not found in solution.")
        return self._solution_values[name]

    def get_all_values(self) -> Dict[str, float]:
        """Returns dictionary of all variable values in the solution."""
        return self._solution_values.copy()

