from rl.algos.bandit import EpsilonKArmedLearner
from rl.world.bandit import StationaryKArmed
from rl.experiments.bandit.viewer.terminal import TerminalKArmedViewer
from rl.experiments.bandit.controller import KArmedController
import trio

if __name__ == '__main__':
    b = StationaryKArmed(k=4)
    m = EpsilonKArmedLearner(
            bandit=b,
            epsilon=0.1,
            alpha=lambda _: 0.1)
    v = TerminalKArmedViewer()
    c = KArmedController(agent=m, 
            views=[v])
    trio.run(c.run_simulation, 10)
 