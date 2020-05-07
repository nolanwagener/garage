"""Exploration strategies which use NumPy as a numerical backend."""
from garage.np.exploration_policies.epsilon_greedy_policy import (
    EpsilonGreedyPolicy)
from garage.np.exploration_policies.exploration_policy import ExplorationPolicy
from garage.np.exploration_policies.gaussian_policy import GaussianPolicy
from garage.np.exploration_policies.ou_policy import OUPolicy

__all__ = [
    'EpsilonGreedyPolicy', 'ExplorationPolicy', 'GaussianPolicy', 'OUPolicy'
]
