from typing import Dict, Any
from rl.experiments.bandit.controller_protocol import KArmedControllerViewerHandle
import urwid

# TODO: use https://github.com/urwid/urwid/pull/298

class TerminalKArmedViewer:
    def open(self):
        print('New k-armed experiment')
        
    def close(self):
        print('End of experiment')
        
    def set_controller(self, c: KArmedControllerViewerHandle):
        pass
        
    async def update(self, state: Dict[str, Any]):
        print(state)
