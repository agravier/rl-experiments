from typing import Dict, Any, Iterable, List
from rl.algos.bandit import KArmedLearner
from rl.experiments.bandit.viewer import KArmedViewer
import trio


class KArmedController:
    """Controller sychronizing a k-armed bandit experiment
    and its viewers, which may be live visualizations, UI or
    recording plugins."""
    agent: KArmedLearner
    views: List[KArmedViewer]
    running: bool = False
    _total_reward: float = 0.
    _step_number: int = 0
    total_steps: int = 0

    def __init__(self, agent: KArmedLearner, 
            views: Iterable[KArmedViewer]) -> None:
        self.agent = agent
        self.views = list(views)
        for v in self.views: 
            v.set_controller(self)
        
    @property
    def state(self) -> Dict[str, Any]:
        """Experiment state information communicated to
        visualizer"""
        state = dict(
            step_number = self._step_number,
            total_steps = self.total_steps,
            action_estimates = self.agent.estimates,
            true_means = self.agent.bandit.means,
            total_reward = self._total_reward,
            running = self.running)
        return state

    async def update_views(self, nursery: 'trio._core.Nursery'):
        for v in self.views:
            nursery.start_soon(v.update, self.state)
     
    async def run_simulation(self, num_steps: int):
        self.total_steps += num_steps
        while self._step_number <= self.total_steps:
            async with trio.open_nursery() as nursery:
                self._step_number += 1
                self.agent.step()
                nursery.start_soon(self.update_views, nursery)

    # viewer handle protocol functions
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
        
