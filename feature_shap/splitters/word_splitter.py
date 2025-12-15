import re
from feature_shap.splitters import SplitterBase


class WordSplitter(SplitterBase):
    """
    A splitter that breaks down a prompt into individual words and punctuation.

    This splitter uses regular expressions to separate words, spaces, and various
    punctuation marks, treating each as a distinct feature. This allows for a
    fine-grained analysis of how each component of the prompt contributes to the
    model's output.
    """
    def __init__(self):
        """
        Initializes the WordSplitter.
        """
        pass

    def split(self, prompt):
        """
        Splits the prompt into a list of words.

        Args:
            prompt (str): The input text to split.

        Returns:
            List[str]: A list of strings, where each element is a word.
        """
        return prompt.split()

    def join(self, parts):
        """
        Joins a list of words back into a single string.

        Args:
            parts (List[str]): The list of strings to join.

        Returns:
            str: The reconstructed string.
        """
        return " ".join(parts)
