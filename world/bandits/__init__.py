"""Data generators for multi-armed bandit experiments"""
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr

class KArmed(object):
	"""k-armed Gaussian bandits"""
	def __init__(self, k: int):
		self._k = k
		
	@property
	def k(self) -> int:
		return self._k
		
if __name__ == '__main__':
	assert KArmed(3).k == 3
	print('OK')