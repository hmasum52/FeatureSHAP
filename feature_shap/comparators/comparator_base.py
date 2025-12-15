from abc import ABC, abstractmethod

class ComparatorBase(ABC):
    """
    Abstract base class for comparators.

    Comparators are responsible for calculating a similarity score between two texts.
    """
    @abstractmethod
    def compare(self, text1: str, text2: str) -> float:
        """
        Computes the similarity between two texts.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: A similarity score, typically between 0.0 and 1.0.
        """
        pass
