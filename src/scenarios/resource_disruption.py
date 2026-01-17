"""Resource Disruption and Absenteeism Scenario Generator."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import pandas as pd

from src.data.schemas import ProblemInstance, Resource


class ResourceDisruptionGenerator:
    """Simulates supply-side shocks, absenteeism, and equipment downtime."""

    def __init__(self, base_instance: ProblemInstance, seed: int = 42):
        self.base_instance = base_instance
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_disruption_scenario(
        self,
        scenario_name: str = "resource_disruption",
        disruption_probability: float = 0.20,
        capacity_loss_ratio: float = 0.50
    ) -> ProblemInstance:
        """Generates a disruption scenario where a fraction of resources suffer capacity degradation."""
        scen_inst = copy.deepcopy(self.base_instance)
        scen_inst.scenario_name = scenario_name
        scen_inst.instance_id = f"{self.base_instance.instance_id}_{scenario_name}"

        disrupted_resources: List[str] = []
        for r_id, res in scen_inst.resources.items():
            if self.rng.random() < disruption_probability:
                # Capacity reduced by capacity_loss_ratio (or set to zero if 1.0)
                new_cap = round(max(0.0, res.capacity_per_period * (1.0 - capacity_loss_ratio)), 2)
                res.capacity_per_period = new_cap
                res.max_overtime_per_period = round(new_cap * 0.25, 2)
                disrupted_resources.append(r_id)

        # Ensure at least 1 resource is disrupted if probability > 0 and resources exist
        if not disrupted_resources and disruption_probability > 0 and scen_inst.resources:
            target_r = str(self.rng.choice(list(scen_inst.resources.keys())))
            res = scen_inst.resources[target_r]
            res.capacity_per_period = round(max(0.0, res.capacity_per_period * (1.0 - capacity_loss_ratio)), 2)
            res.max_overtime_per_period = round(res.capacity_per_period * 0.25, 2)
            disrupted_resources.append(target_r)

        scen_inst.metadata["disrupted_resources"] = disrupted_resources
        scen_inst.metadata["disruption_probability"] = disruption_probability
        return scen_inst

    def generate_combined_scenario(
        self,
        scenario_name: str = "combined_disruption",
        demand_multiplier: float = 1.25,
        disruption_probability: float = 0.20
    ) -> ProblemInstance:
        """Combines increased demand with reduced resource availability."""
        scen_inst = self.generate_disruption_scenario(
            scenario_name=scenario_name,
            disruption_probability=disruption_probability,
            capacity_loss_ratio=0.50
        )

        for task in scen_inst.tasks.values():
            task.resource_demand = round(max(0.5, task.resource_demand * demand_multiplier), 2)

        scen_inst.metadata["demand_multiplier"] = demand_multiplier
        return scen_inst

    def generate_standard_disruptions(self) -> Dict[str, ProblemInstance]:
        """Returns standard Scenario C (Disruption) and Scenario D (Combined)."""
        return {
            "resource_disruption": self.generate_disruption_scenario("resource_disruption", disruption_probability=0.20),
            "combined_disruption": self.generate_combined_scenario("combined_disruption", demand_multiplier=1.25, disruption_probability=0.20)
        }

    def export_disruptions_csv(
        self,
        scenarios: Dict[str, ProblemInstance],
        output_dir: Path | str
    ) -> Dict[str, Path]:
        """Exports resource disruption profiles into CSV."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        files: Dict[str, Path] = {}

        for name, inst in scenarios.items():
            records = [
                {
                    "resource_id": r_id,
                    "name": res.name,
                    "skills": ";".join(res.skills),
                    "capacity_per_period": res.capacity_per_period,
                    "max_overtime_per_period": res.max_overtime_per_period,
                    "regular_cost": res.regular_cost_per_period
                }
                for r_id, res in inst.resources.items()
            ]
            df = pd.DataFrame.from_records(records)
            csv_file = out_path / f"{name}_capacity.csv"
            df.to_csv(csv_file, index=False)
            files[name] = csv_file

        return files

