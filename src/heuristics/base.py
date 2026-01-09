"""Base class for metaheuristic scheduling algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

from src.data.schemas import AllocationSolution, ProblemInstance
from src.data.config_loader import ModelConfig, load_model_config


class BaseMetaheuristic(ABC):
    """Abstract base class for heuristic and metaheuristic solvers."""

    def __init__(
        self,
        instance: ProblemInstance,
        config: Optional[ModelConfig] = None,
        seed: int = 42
    ):
        self.instance = instance
        self.config = config or load_model_config()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_solution: Optional[AllocationSolution] = None
        self.best_cost: float = float("inf")
        self.convergence_history: List[Dict[str, Any]] = []

    @abstractmethod
    def solve(self, **kwargs) -> AllocationSolution:
        """Executes the metaheuristic optimization search."""
        pass

