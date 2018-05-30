from typing import Dict, Any
from rl.algos.bandit import KArmedLearner
from rl.experiments.bandit.viewer import KArmedViewer

class KArmedController():
    """Controller sychronizing a k-armed bandit experiment
    and its live visualization."""
    agent: KArmedLearner
    view: KArmedViewer
    running: bool = False
    _total_reward: float = 0.
    _step_number: int = 0

    def __init__(self,
            agent: KArmedLearner, view: KArmedViewer) -> None:
        self.agent = agent
        self.view = view
        
    @property
    def state(self) -> Dict[str, Any]:
        """Experiment state information communicated to
        visualizer"""
        state = dict(
            step_number = self._step_number,
            action_estimates = self.agent.estimates,
            true_means = self.agent.bandit.means,
            total_reward = self._total_reward,
            running = self.running)
        return state

    def open_view(self):
        self.view.open()

    def close_view(self):
        self.view.close()
        
    def update_view(self):
        self.view.update(self.state)

    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
 