from feature_shap.comparators.comparator_base import ComparatorBase
import evaluate


class BertScoreComparator(ComparatorBase):
    """
    A comparator that uses BERTScore to calculate similarity between two texts.

    BERTScore computes a similarity score by aligning tokens in the candidate and reference
    texts and calculating their cosine similarity using contextual embeddings.
    """
    def __init__(self, lang: str = "en"):
        """
        Initialize the BERTScore comparator.

        Args:
            lang (str, optional): The language of the texts. Defaults to "en".
        """
        self.lang = lang
        self.bertscore_metric = evaluate.load("bertscore")


    def compare(self, text1: str, text2: str) -> float:
        """
        Compute the BERTScore F1 between two texts.

        Args:
            text1 (str): The candidate text.
            text2 (str): The reference text.

        Returns:
            float: The BERTScore F1 score, ranging from 0.0 to 1.0.
        """
        if not text1 or not text2:
            return 0.0

        result = self.bertscore_metric.compute(
            predictions=[text1],
            references=[text2],
            lang=self.lang
        )

        return float(result["f1"][0])
