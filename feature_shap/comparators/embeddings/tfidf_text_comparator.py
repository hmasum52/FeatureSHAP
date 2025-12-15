from feature_shap.comparators.comparator_base import ComparatorBase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfTextComparator(ComparatorBase):
    """
    A comparator that uses TF-IDF vectors to calculate text similarity.

    This class represents texts as TF-IDF vectors and then computes the cosine
    similarity between them.
    """
    def __init__(self, corpus: list = None):
        """
        Initialize the TF-IDF vectorizer.

        Args:
            corpus (list, optional): A list of reference texts to fit the TF-IDF model.
        """
        self.vectorizer = TfidfVectorizer()
        if corpus:
            self.vectorizer.fit(corpus)  # Pre-fit if corpus is provided


    def vectorize(self, text: str) -> np.ndarray:
        """
        Generates a single TF-IDF vector for one text.

        Args:
            text (str): The text to vectorize.

        Returns:
            np.ndarray: The TF-IDF vector.
        
        Raises:
            ValueError: If the TF-IDF model is not trained.
        """
        if self.vectorizer.vocabulary_ is None:
            raise ValueError("TF-IDF model is not trained. Provide a corpus at initialization or train with vectorize().")

        vector = self.vectorizer.transform([text]).toarray()
        return vector[0]  # Return a single vector


    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Computes the cosine similarity between two texts and scales it to a [0, 1] range.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The scaled similarity score.
        """
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)

        # Compute cosine similarity (reshaping required for sklearn function)
        similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        
        # Scale from [-1, 1] to [0, 1]
        return (float(similarity) + 1) / 2

    def compare(self, text1, text2):
        """
        Compares two texts by calculating their similarity.

        Args:
            text1 (str): The first text.
            text2 (str): The second text.

        Returns:
            float: The similarity score.
        """
        return self.calculate_similarity(text1, text2)
