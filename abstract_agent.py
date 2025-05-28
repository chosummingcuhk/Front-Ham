from abc import ABC


class AbstractAgent(ABC):
    def __init__(self, color ):
        self.color = color
    def action(self, state):
        pass

