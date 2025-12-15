from feature_shap.comparators.comparator_base import ComparatorBase
import evaluate


class BLEUComparator(ComparatorBase):
    """
    A comparator that uses the BLEU score to measure the similarity between two texts.

    BLEU (Bilingual Evaluation Understudy) is a metric for evaluating a generated sentence
    to a reference sentence.
    """
    def __init__(self):
        """
        Initialize the BLEU comparator using Hugging Face's evaluate library.
        """
        self.bleu_metric = evaluate.load("bleu")


    def compare(self, text1: str, text2: str) -> float:
        """
        Compute the BLEU score between two texts.

        Args:
            text1 (str): The candidate text (e.g., the generated text).
            text2 (str): The reference text.

        Returns:
            float: The BLEU score, ranging from 0.0 to 1.0.
        """
        if not text1 or not text2:
            return 0.0

        # Compute BLEU using text1 as hypothesis and text2 as reference
        result = self.bleu_metric.compute(predictions=[text1], references=[[text2]])

        return float(result["bleu"])
