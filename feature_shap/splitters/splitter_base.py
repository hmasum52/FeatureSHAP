from abc import ABC, abstractmethod


class SplitterBase(ABC):
    """
    Abstract base class for prompt splitters.

    Splitters are responsible for dividing a prompt into a list of features (strings).
    """
    @abstractmethod
    def split(self, prompt):
        """
        Splits a prompt into a list of features.

        Args:
            prompt (str): The prompt to split.

        Returns:
            list: A list of features (strings).
        """
        pass


    @abstractmethod
    def join(self, parts):
        """
        Joins a list of features back into a single string.

        Args:
            parts (list): A list of features (strings).

        Returns:
            str: The joined string.
        """
        pass
