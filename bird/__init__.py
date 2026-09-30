"""BIRD -- Benchmark for Iterative Reward Design.

One algorithm, one hyperparameter space. Every published LLM-driven
reward-design method is a point in that space, selected by a YAML config.

Every key and its allowed values are declared in `configs/_default.yaml` and
`bird/schema.py`.
"""

__version__ = "0.1.0"

from .config import Config, ConfigError, load  # noqa: F401
from .context import Context  # noqa: F401
from .state import RunState  # noqa: F401
from .types import (  # noqa: F401
    Candidate,
    CandidateReport,
    Preference,
    Selection,
    Trajectory,
    TrainResult,
)

__all__ = [
    "Config", "ConfigError", "load", "Context", "RunState",
    "Candidate", "CandidateReport", "Preference", "Selection",
    "Trajectory", "TrainResult",
]
