"""Configuration Loader and Pydantic Schemas for Resource Allocation Optimization."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class SolverConfig(BaseModel):
    """HiGHS / MILP Solver Configuration Settings."""
    name: str = "highs"
    time_limit_seconds: float = Field(default=60.0, ge=0.1)
    mip_rel_gap: float = Field(default=1e-4, ge=0.0)
    mip_abs_gap: float = Field(default=1e-6, ge=0.0)
    log_to_console: bool = True
    presolve: str = "on"
    parallel_threads: int = Field(default=4, ge=1)


class CostConfig(BaseModel):
    """Cost weights and multipliers."""
    regular_cost_multiplier: float = Field(default=1.0, ge=0.0)
    overtime_cost_multiplier: float = Field(default=1.5, ge=0.0)
    idle_cost_multiplier: float = Field(default=0.25, ge=0.0)
    unmet_demand_penalty: float = Field(default=1000.0, ge=0.0)


class ConstraintConfig(BaseModel):
    """Operational and scheduling constraint toggles and parameters."""
    allow_overtime: bool = True
    max_overtime_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    enforce_skill_compatibility: bool = True
    enforce_precedence: bool = True
    min_rest_periods: int = Field(default=1, ge=0)
    max_consecutive_periods: int = Field(default=5, ge=1)


class MetaheuristicConfig(BaseModel):
    """Hyperparameters for Genetic Algorithm, Simulated Annealing, etc."""
    population_size: int = Field(default=50, ge=4)
    generations: int = Field(default=100, ge=1)
    mutation_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.80, ge=0.0, le=1.0)
    tournament_size: int = Field(default=3, ge=1)
    sa_initial_temp: float = Field(default=1000.0, gt=0.0)
    sa_cooling_rate: float = Field(default=0.95, gt=0.0, lt=1.0)
    sa_min_temp: float = Field(default=0.01, gt=0.0)


class ModelConfig(BaseModel):
    """Combined optimization model configuration."""
    costs: CostConfig = Field(default_factory=CostConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    metaheuristics: MetaheuristicConfig = Field(default_factory=MetaheuristicConfig)


class ScaleConfig(BaseModel):
    """Instance scale specifications."""
    n_resources: int = Field(..., ge=1)
    n_tasks: int = Field(..., ge=1)
    n_periods: int = Field(..., ge=1)
    description: str = ""


class ScenarioConfig(BaseModel):
    """Uncertainty scenario specifications."""
    demand_multiplier: float = Field(default=1.0, ge=0.0)
    disruption_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""


class ExperimentMetadata(BaseModel):
    name: str = "multi_period_benchmark"
    random_seed: int = 42
    num_replications: int = 5


class ExperimentConfig(BaseModel):
    """Experiment benchmark suite configuration."""
    experiment: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    scales: Dict[str, ScaleConfig]
    scenarios: Dict[str, ScenarioConfig]


def get_project_root() -> Path:
    """Returns the root directory of the project repository."""
    # src/data/config_loader.py -> parent x 2 = project root
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Safely loads a YAML file into a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_solver_config(config_path: Optional[Path | str] = None) -> SolverConfig:
    """Loads and validates solver_config.yaml."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "solver_config.yaml"
    data = load_yaml(config_path)
    solver_data = data.get("solver", data)
    return SolverConfig(**solver_data)


def load_model_config(config_path: Optional[Path | str] = None) -> ModelConfig:
    """Loads and validates model_config.yaml."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "model_config.yaml"
    data = load_yaml(config_path)
    return ModelConfig(**data)


def load_experiment_config(config_path: Optional[Path | str] = None) -> ExperimentConfig:
    """Loads and validates experiment_config.yaml."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "experiment_config.yaml"
    data = load_yaml(config_path)
    return ExperimentConfig(**data)


def load_configs(config_dir: Optional[Path | str] = None) -> tuple[SolverConfig, ModelConfig, ExperimentConfig]:
    """Loads all three core configurations at once."""
    base_dir = Path(config_dir) if config_dir else (get_project_root() / "configs")
    s_cfg = load_solver_config(base_dir / "solver_config.yaml")
    m_cfg = load_model_config(base_dir / "model_config.yaml")
    e_cfg = load_experiment_config(base_dir / "experiment_config.yaml")
    return s_cfg, m_cfg, e_cfg

