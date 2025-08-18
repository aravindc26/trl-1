from typing_extensions import Optional, Dict, Any, Type, TypeVar
from dataclasses import field
from ..trainer.grpo_config import GRPOConfig

T = TypeVar("T")

class MultiTurnGRPOConfig(GRPOConfig):
    env_class: Optional[Type[T]] = field(
        default=None,
        metadata={
            "help": "Environment class to use for training. If not provided, the environment will be inferred from the "
            "model's configuration."
        },
    )

    env_init_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for the environment constructor."
        },
    )

    max_turns: int = field(
        default=10,
        metadata={
            "help": "Maximum number of turns to generate."
        },
    )