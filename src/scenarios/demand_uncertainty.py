"""Demand Uncertainty and Fluctuation Scenario Generator."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import pandas as pd

from src.data.schemas import ProblemInstance, Task


class DemandScenarioGenerator:
    """Generates stochastic demand realizations and predefined operational scenarios."""

    def __init__(self, base_instance: ProblemInstance, seed: int = 42):
        self.base_instance = base_instance
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_scaled_scenario(
        self,
        scenario_name: str,
        demand_multiplier: float = 1.0,
        task_subset: Optional[List[str]] = None
    ) -> ProblemInstance:
        """Generates a scenario where task demand is scaled by a multiplier."""
        scen_inst = copy.deepcopy(self.base_instance)
        scen_inst.scenario_name = scenario_name
        scen_inst.instance_id = f"{self.base_instance.instance_id}_{scenario_name}"

        for t_id, task in scen_inst.tasks.items():
            if task_subset is None or t_id in task_subset:
                new_demand = round(max(0.5, task.resource_demand * demand_multiplier), 2)
                task.resource_demand = new_demand

        scen_inst.metadata["demand_multiplier"] = demand_multiplier
        return scen_inst

    def generate_stochastic_scenarios(
        self,
        n_scenarios: int = 5,
        cv: float = 0.20,  # coefficient of variation
        distribution: str = "truncated_normal"
    ) -> List[ProblemInstance]:
        """Generates N Monte Carlo stochastic demand realizations."""
        scenarios: List[ProblemInstance] = []

        for s_idx in range(n_scenarios):
            scen_name = f"stochastic_s{s_idx+1}"
            scen_inst = copy.deepcopy(self.base_instance)
            scen_inst.scenario_name = scen_name
            scen_inst.instance_id = f"{self.base_instance.instance_id}_{scen_name}"

            for t_id, task in scen_inst.tasks.items():
                mean_demand = task.resource_demand
                std_demand = mean_demand * cv

                if distribution == "lognormal":
                    mu = np.log((mean_demand ** 2) / np.sqrt(std_demand ** 2 + mean_demand ** 2))
                    sigma = np.sqrt(np.log(1 + (std_demand ** 2) / (mean_demand ** 2)))
                    sample = float(self.rng.lognormal(mu, sigma))
                else:
                    # Truncated normal between 0.4 and 2.5 * mean
                    raw_sample = float(self.rng.normal(mean_demand, std_demand))
                    sample = max(mean_demand * 0.4, min(mean_demand * 2.5, raw_sample))

                task.resource_demand = round(max(0.5, sample), 2)

            scen_inst.metadata["scenario_index"] = s_idx + 1
            scenarios.append(scen_inst)

        return scenarios

    def generate_standard_scenarios(self) -> Dict[str, ProblemInstance]:
        """Generates the standardized benchmark scenario suite."""
        return {
            "baseline": self.generate_scaled_scenario("baseline", demand_multiplier=1.0),
            "high_demand": self.generate_scaled_scenario("high_demand", demand_multiplier=1.25),
            "low_demand": self.generate_scaled_scenario("low_demand", demand_multiplier=0.75),
            "peak_demand": self.generate_scaled_scenario("peak_demand", demand_multiplier=1.50),
        }

    def export_scenarios_csv(
        self,
        scenarios: Dict[str, ProblemInstance],
        output_dir: Path | str
    ) -> Dict[str, Path]:
        """Exports demand scenarios as CSV tables into output_dir."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        files: Dict[str, Path] = {}

        for name, inst in scenarios.items():
            records = [
                {
                    "task_id": t_id,
                    "task_name": task.name,
                    "required_skill": task.required_skill,
                    "duration": task.duration,
                    "resource_demand": task.resource_demand,
                    "priority": task.priority,
                    "unmet_penalty": task.unmet_penalty
                }
                for t_id, task in inst.tasks.items()
            ]
            df = pd.DataFrame.from_records(records)
            csv_file = out_path / f"{name}_demand.csv"
            df.to_csv(csv_file, index=False)
            files[name] = csv_file

        return files

