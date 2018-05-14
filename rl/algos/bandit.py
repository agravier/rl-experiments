#pylint:disable=E0611
#pylint:disable=C0411
"""This module implements action-value methods
for non-associative settings such as k-armed bandit
problems: epsilon-greedy with sample average or
exponential recency-weighted average, or any arbirary
step size function."""
from typing import Callable, Union
from types import MethodType
from collections import  Counter
from rl.world.bandits import KArmed

class EpsilonKArmedLearner:
    """Epsilon-greedy k-armed bandit solver with action
    value estimates incrementally updated from samples
    with the given step-size function alpha. Parameter alpha
    is a function of the number of times the action has been
    selected, returning the StepSize parameter of the
    incremental expression of the action value estimate:
    NewEstimate ← OldEstimate + StepSize [Target- OldEstimate]
    (Barto & Sutton)."""
    actions_count: Counter
    _epsilon: Callable[[int], float]
    _alpha: Callable[[int], float]
    bandit: KArmed
    
    def __init__(
            self, bandit: KArmed,
            epsilon: Union[float, Callable[[int], float]],
            alpha: Callable[[int], float]) -> None:
        if isinstance(epsilon, float):
            assert 0 <= epsilon <= 1
            self._epsilon = lambda _: epsilon  # type: ignore
        else:
            self._epsilon = epsilon  # type: ignore
        self._alpha = alpha  # type: ignore
        self.actions_count = Counter()
        self.bandit = bandit
        
    @property
    def epsilon(self) -> float:
        return self._epsilon(sum(self.actions_count))  # type: ignore

    def alpha(self, k: int) -> float:
        n = self.actions_count.get(k, 0)
        assert n > 0, 'Step size is undefined for actions that were never taken'
        return self._alpha(n)  # type: ignore
