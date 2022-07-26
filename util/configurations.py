import argparse
from abc import ABC
from typing import Any, Optional, Type, TypeVar, Union

from util.io import get_runname


def parse_args() -> argparse.Namespace:
    """Parses the programme arguments."""

    # construct parser
    parser = argparse.ArgumentParser(
        description="Launches simulation for different MPC-based RL agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("Agent type (select only one)")
    group.add_argument(
        "--lstdq",
        action="store_true",
        help="If passed, trains the LSTD Q-learning agent WITHOUT the safety "
        "mechanism.",
    )
    group.add_argument(
        "--safe-lstdq",
        "--safe_lstdq",
        action="store_true",
        help="If passed, trains the LSTD Q-learning agent WITH the safety "
        "mechanism.",
    )
    group.add_argument(
        "--pk",
        action="store_true",
        help="If passed, evaluates a non-learning PK agent.",
    )

    group = parser.add_argument_group("RL algorithm parameters")
    group.add_argument("--gamma", type=float, default=0.9792, help="Discount factor.")
    group.add_argument(  # [3e-2, 3e-2, 1e-3, 1e-3, 1e-3],
        "--lr",
        type=float,
        nargs="+",
        default=[0.498],
        help="Learning rate. Can be a single float, or a list of floats. In "
        "the latter case, either one float per parameter name, or one "
        "per parameter element (in case of parameters that are vectors).",
    )
    group.add_argument(
        "--perturbation-decay",
        "--perturbation_decay",
        type=float,
        default=0.885,
        help="Rate at which the exploration (random perturbations of the MPC "
        "objective ) decays both in term of chance and strength.",
    )
    group.add_argument(
        "--max-perc-update",
        "--max_perc_update",
        type=float,
        default=float("inf"),
        help="Limits the maximum value that each parameter can be updated by "
        "a percentage of the current value.",
    )

    group = parser.add_argument_group("Simulation length")
    group.add_argument(
        "--agents", type=int, default=100, help="Number of parallel agent to simulate."
    )
    group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per each agent.",
    )
    group.add_argument(
        "--episodes", type=int, default=1, help="Number of training episodes per epoch."
    )
    group.add_argument(
        "--max_ep_steps",
        type=int,
        default=50,
        help="Maximum number of steps per training episode.",
    )

    group = parser.add_argument_group("Simulation details")
    group.add_argument(
        "--runname", type=str, default=None, help="Name of the simulation run."
    )
    group.add_argument("--seed", type=int, default=1909, help="RNG seed.")
    group.add_argument(
        "--n-jobs", "--n_jobs", type=int, default=-1, help="Simulation parallel jobs."
    )
    group.add_argument("--verbose", action="store_true", help="Verbose flag.")

    group = parser.add_argument_group("RL experience replay parameters")
    group.add_argument(
        "--replay-mem-size",
        "--replay_mem_size",
        type=int,
        default=1,
        help="How many epochs the replay memory can store.",
    )
    group.add_argument(
        "--replay-mem-sample",
        "--replay_mem_sample",
        type=float,
        default=1.0,
        help="Size of the replay memory samples (percentage).",
    )

    group = parser.add_argument_group("GP parameters (used only for safe algorithm)")
    group.add_argument(
        "--gp-alpha",
        "--gp_alpha",
        type=float,
        default=1e-10,
        help="Estimated measurement noise of the GP data.",
    )
    group.add_argument(
        "--gp-kernel",
        "--gp_kernel",
        choices=("RBF", "Matern"),
        default="RBF",
        help="Type of GP kernel.",
    )
    group.add_argument(
        "--average-violation",
        "--average_violation",
        action="store_true",
        help="Attempts to reduce GP training data by averaging violations "
        "over episodes in the same epoch.",
    )
    group.add_argument(
        "--prior",
        type=str,
        default=None,
        help="Loads prior knowledge on safety from the specified simulation "
        "data (must be a pickled file of safe-lstdq agents)",
    )

    args = parser.parse_args()

    # perform some checks
    assert (
        args.lstdq + args.safe_lstdq + args.pk == 1
    ), "Must specify one and only one type of agent to simulate."
    if args.prior is not None:
        assert (
            args.safe_lstdq
        ), "Prior safety knowledge can only be specified for safe algorithm"

    # set some defaults
    if args.safe_lstdq and (args.n_jobs == -1 or args.n_jobs > 1):
        import os

        os.environ["PYTHONWARNINGS"] = "ignore"  # ignore warnings
    if args.agents == 1:
        args.n_jobs = 1  # don't parallelize
    args.runname = get_runname(candidate=args.runname)

    return args


class BaseConfig(ABC):
    """Base abstract class for configurations."""

    def get_group(self, group: str) -> dict[str, Any]:
        """Gets a group of parameters starting wit the name `group_`, where `group` is
        the given string."""
        return {
            name.removeprefix(f"{group}_"): val
            for name, val in self.__dict__.items()
            if name.startswith(f"{group}_")
        }


ConfigType = TypeVar("ConfigType", bound=BaseConfig)


def init_config(
    config: Optional[Union[ConfigType, dict]], cls: Type[ConfigType]
) -> ConfigType:
    """
    Initializes the configuration, by
        - returning it, if valid
        - converting from a dict to a dataclass, if a dict is provided
        - instantiating the default configuration, if None is passed.
    """
    if config is None:
        return cls()

    if isinstance(config, cls):
        return config

    if isinstance(config, dict):
        if not hasattr(cls, "__dataclass_fields__"):
            raise ValueError("Configiration class must be a dataclass.")
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: config[k] for k in keys if k in config})

    raise ValueError(
        "Invalid configuration type; expected None, dict or "
        f"a dataclass, got {cls} instead."
    )
