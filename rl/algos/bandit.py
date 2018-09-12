"""This module implements action-value methods
for non-associative settings such as k-armed bandit
problems: epsilon-greedy with sample average or
exponential recency-weighted average, or any arbirary
step size function."""
import random
import sys
from operator import itemgetter
from typing import Callable, Union, List, Tuple
from collections import  Counter
from typing_extensions import Protocol
from rl.world.bandit import KArmed


_IndexedEstimate = Tuple[float, int]


class KArmedLearner(Protocol):
    """Protocol for k-armed RL algos"""
    bandit: KArmed
    def step(self) -> Tuple[int, float]: ...
    @property
    def estimates(self) -> List[float]: ...


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
    _estimates: List[_IndexedEstimate]
    _epsilon: Callable[[int], float]
    _alpha: Callable[[int], float]
    bandit: KArmed
    _rng: random.Random
    _seed: int
    _k: int
    
    def __init__(
            self, bandit: KArmed,
            epsilon: Union[float, Callable[[int], float]],
            alpha: Callable[[int], float],
            initializer: Callable[[int], float] = None,
            seed: int = None) -> None:
        # Create and store random seed
        if seed is None:
            seed = random.randrange(sys.maxsize)
        self._rng = random.Random(seed)
        self._seed = seed
        # Initialize value estimates
        if initializer is None:
            initializer = lambda _: 0
        self._estimates = [(initializer(i), i) for i in range(bandit.k)]
        self._estimates.sort()
        # Store exploration ratio function
        if isinstance(epsilon, float):
            assert 0 <= epsilon <= 1
            self._epsilon = lambda _: epsilon  # type: ignore
        else:
            self._epsilon = epsilon  # type: ignore
        # Store step size, actions counter and bandit
        self._alpha = alpha  # type: ignore
        self.actions_count = Counter()
        self.bandit = bandit
        self._k = self.bandit.k
        
    @property
    def epsilon(self) -> float:
        return self._epsilon(sum(self.actions_count))  # type: ignore

    def alpha(self, k: int) -> float:
        n = self.actions_count.get(k, 0)
        assert n > 0, 'Step size is undefined for actions that were never taken'
        return self._alpha(n)  # type: ignore
    
    def step(self) -> Tuple[int, float]:
        """Take an exploit/explore decision with the probability
        to explore determined by epsilon. Update the
        estimated value of the selected action, moving it
        towards its latest reward with a step size of alpha."""
        draw = self._rng.uniform(0, 1)
        if draw <= self.epsilon:
            idx = self._rng.choice(range(self._k))
        else:
            idx = -1
        q_a, a = self._estimates[idx]
        reward = self.bandit.pull_lever(a)
        self.actions_count[a] += 1
        delta = reward - q_a
        alpha = self.alpha(a)
        new_q_a = q_a + alpha * delta
        self._estimates[idx] = new_q_a, a
        self._estimates.sort()
        return a, reward

    @property
    def estimates(self) -> List[float]:
        by_idx = sorted(self._estimates, key=itemgetter(1))
        return [v for v, _ in by_idx]