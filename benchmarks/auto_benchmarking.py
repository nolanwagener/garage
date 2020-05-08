#!/usr/bin/env python3
"""Run automatic benchmarking."""
# pylint: disable=wrong-import-order
import warnings

import click

# pylint: disable=unused-import
import garage.config  # noqa: F401

from benchmarks.experiments.algos import ddpg_garage_tf  # noqa: I100
from benchmarks.experiments.algos import ppo_garage_pytorch
from benchmarks.experiments.algos import ppo_garage_tf
from benchmarks.experiments.algos import td3_garage_tf
from benchmarks.experiments.algos import trpo_garage_pytorch
from benchmarks.experiments.algos import trpo_garage_tf
from benchmarks.experiments.algos import vpg_garage_pytorch
from benchmarks.experiments.algos import vpg_garage_tf
from benchmarks.helper import benchmark, iterate_experiments
from benchmarks.parameters import MuJoCo1M_ENV_SET


@benchmark(plot=False, auto=True)
def auto_ddpg_benchmarks():
    """Run experiments for DDPG benchmarking."""
    iterate_experiments(ddpg_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_ppo_benchmarks():
    """Run experiments for PPO benchmarking."""
    iterate_experiments(ppo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(ppo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_td3_benchmarks():
    """Run experiments for TD3 benchmarking."""
    td3_env_ids = [
        env_id for env_id in MuJoCo1M_ENV_SET if env_id != 'Reacher-v2'
    ]

    iterate_experiments(td3_garage_tf, td3_env_ids)


@benchmark(plot=False, auto=True)
def auto_trpo_benchmarks():
    """Run experiments for TRPO benchmarking."""
    iterate_experiments(trpo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(trpo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark(plot=False, auto=True)
def auto_vpg_benchmarks():
    """Run experiments for VPG benchmarking."""
    iterate_experiments(vpg_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(vpg_garage_tf, MuJoCo1M_ENV_SET)


@click.command()
@click.argument('runs')
def auto_benchmarking(runs):
    """Run automatic benchmarking.

    Args:
        runs (list): A list of automatic benchmark functions to run.
             Function name must be defined in this file with benchmark
             decorator and auto=True.

    """
    benchmarks = [
        auto_ddpg_benchmarks, auto_ppo_benchmarks, auto_td3_benchmarks,
        auto_trpo_benchmarks, auto_vpg_benchmarks
    ]
    runs = [b for b in benchmarks if b.__name__ in runs]

    if not runs:
        warnings.warn('No benchmark to run! Make sure you pass a list'
                      'of function names and they match the function'
                      'names defined in auto_benchmarking.py')
    else:
        for func in runs:
            func()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    auto_benchmarking()
