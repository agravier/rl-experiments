from rl.algos.bandit import KArmedLearner
from rl.experiments.bandit.viewer import KArmedViewer

class KArmedController():
    agent: KArmedLearner
    view: KArmedViewer

    def __init__(self,
            agent: KArmedLearner, view: KArmedViewer) -> None:
        self.agent = agent
        self.view = view

    def run(self):
        self.view.create()
        
    def view_arms(self):