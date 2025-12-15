from feature_shap.splitters import SplitterBase
import re

class PointsSplitter(SplitterBase):
    """
    A splitter that divides a text into segments based on punctuation marks,
    specifically periods.
    """
    def __init__(self):
        """
        Initializes the PointsSplitter.
        """
        self.split_pattern = r'[^.]*\.(?:\s*|$)|[^.]+$'

    def split(self, prompt):
        """
        Splits a prompt into segments at each period.

        Args:
            prompt (str): The text to split.

        Returns:
            list: A list of text segments.
        """
        return re.findall(self.split_pattern, prompt)

    def join(self, parts):
        """
        Joins a list of parts back into a single string.

        Args:
            parts (list): A list of text segments.

        Returns:
            str: The joined string.
        """
        return ''.join(parts)
