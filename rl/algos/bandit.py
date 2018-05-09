"""This module implements action-value methods
for non-associative settings such as k-armed bandit
problems: epsilon-greedy with sample average or
exponential recency-weighted average, or any arbirary
step size function."""

class EpsilonKArmedLearner(object):
    """Epsilon-greedy k-armed bandit solver with action
    value estimates incrementally updated from samples
    with the given step-size function a"""
    def __init__(self, bandit: KArmed)