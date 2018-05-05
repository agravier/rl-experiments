"""Data generators for multi-armed bandit experiments"""
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr
import random
import sys


class KArmed(object):
    """k-armed Gaussian bandit with unit variances and
    mean rewards drawn from N(0, 1)"""
    def __init__(self, k: int, seed: int = None):
        self._k = k
        if seed is None:
            seed = random.randrange(sys.maxsize)
        self._rng = random.Random(seed)
        self._seed = seed
        self._means = tuple(
            self._rng.gauss(0, 1) for _ in range(k))
		
    @property
    def k(self) -> int:
        return self._k
        
    @property
    def means(self):
        return self._means
    
    def pull_lever(self, i: int) -> float:
        return self._rng.gauss(self._means[i], 1)


if __name__ == '__main__':
    print('Running self-tests')
    BANDIT = KArmed(3)
    assert BANDIT.k == 3
    SAMPLE_MEANS = tuple(
        sum(BANDIT.pull_lever(i) for _ in range(1000)) /1000
            for i in range(BANDIT.k))
    assert all(abs(BANDIT.means[j]-SAMPLE_MEANS[j])<0.5
        for j in range(BANDIT.k))
    print('OK')
    