from feature_shap.comparators.comparator_base import ComparatorBase
from codebleu import calc_codebleu
from typing import Tuple


class CodeBLEUComparator(ComparatorBase):
    """
    A comparator that uses the CodeBLEU score to measure the similarity between two code snippets.

    CodeBLEU is an extension of BLEU that is specifically designed for evaluating code.
    It considers not only n-gram matches but also syntactic and semantic properties of the code.
    """
    def __init__(self,
                 language: str,
                 weights: Tuple[float, float, float, float] = None):
        """
        Initialize the CodeBLEU comparator.

        Args:
            language (str): The programming language of the code snippets.
            weights (Tuple[float, float, float, float], optional): The weights for the different components of the CodeBLEU score.
        """
        self.language = language
        self.weights = weights if weights else (0.25, 0.25, 0.25, 0.25)


    def compare(self, code1: str, code2: str) -> float:
        """
        Compute the CodeBLEU score between two code snippets.

        Args:
            code1 (str): The reference code snippet.
            code2 (str): The candidate code snippet.

        Returns:
            float: The CodeBLEU score, ranging from 0.0 to 1.0.
        """
        if not code1 or not code2:
            return 0.0

        result = calc_codebleu(references=[code1], predictions=[code2],
                               lang=self.language, weights=self.weights)

        return float(result["codebleu"])
