"""Data generators for multi-armed bandit experiments"""
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr
import random
import sys
from typing_extensions import Protocol

class KArmed(Protocol):
    """Interface for k-armed bandit problems"""
    @property
    def k(self) -> int: ...
        
    @property
    def means(self) -> Tuple[float]: ...
    
    def pull_lever(self, i: int) -> float: ...


class StationaryKArmed:
    """k-armed Gaussian bandit with unit variances and
    mean rewards drawn from N(0, 1)"""
    def __init__(self, k: int, seed: int = None) -> None:
        self._k = k
        if seed is None:
            seed = random.randrange(sys.maxsize)
        self._rng = random.Random(seed)
        self._seed = seed
        self._means = tuple(
            self._rng.gauss(0, 1) for _ in range(k))
		
    @property
    def k(self) -> int:
        """The number of levers"""
        return self._k
        
    @property
    def means(self) -> Tuple[float, ...]:
        """The true means of all levers' underlying
        Gaussian RV"""
        return self._means
    
    def pull_lever(self, i: int) -> float:
        """Draw from lever i"""
        return self._rng.gauss(self._means[i], 1)
    