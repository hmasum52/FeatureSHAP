from feature_shap.splitters import SplitterBase


class TokenizerSplitter(SplitterBase):
    """
    A splitter that uses a tokenizer to divide a text into tokens.

    Each token is considered a separate feature.
    """
    def __init__(self, tokenizer):
        """
        Initializes the TokenizerSplitter.

        Args:
            tokenizer: A tokenizer object (e.g., from the `transformers` library).
        """
        self.tokenizer = tokenizer

    def split(self, prompt):
        """
        Splits a prompt into tokens using the provided tokenizer.

        Args:
            prompt (str): The text to split.

        Returns:
            list: A list of tokens.
        """
        return self.tokenizer.tokenize(prompt)

    def join(self, parts):
        """
        Joins a list of tokens back into a single string.

        Args:
            parts (list): A list of tokens.

        Returns:
            str: The joined string.
        """
        return self.tokenizer.convert_tokens_to_string(parts)
