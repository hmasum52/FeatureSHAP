from abc import ABC, abstractmethod
from typing import List, Union


class ModelBase(ABC):
    """
    Abstract base class for language models.

    This class defines the interface for a text generation model.
    """
    @abstractmethod
    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generates text from a given prompt or batch of prompts.

        Args:
            batch (Union[str, List[str]]): A single prompt or a list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or a list of generated texts.
        """
        pass
