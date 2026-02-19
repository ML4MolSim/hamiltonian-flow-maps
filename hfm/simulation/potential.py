from abc import ABC, abstractmethod

class Potential(ABC):
    def __init__(self, masses):
        self.masses = masses

    @abstractmethod
    def compute_force(self, x, p):
        pass

    @abstractmethod
    def compute_epot(self, x, p):
        pass

    @abstractmethod
    def compute_force_and_epot(self, x, p):
        pass

    def compute_velocity(self, x, p):
        m = self.masses

        if len(p.shape) == 3:
            m = m.reshape((1, -1, 1))

        return p / m
