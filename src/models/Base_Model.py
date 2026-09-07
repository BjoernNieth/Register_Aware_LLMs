from abc import ABC, abstractmethod

class Base_Model(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass