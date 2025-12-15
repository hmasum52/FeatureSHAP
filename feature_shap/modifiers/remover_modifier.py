from feature_shap.modifiers.modifier_base import ModifierBase
import re


class RemoverModifier(ModifierBase):
    """
    A modifier that removes non-structural tokens from a feature.

    This is similar to the MaskerModifier, but instead of replacing non-structural
    tokens with a mask, it removes them completely.
    """
    def __init__(self, language="python"):
        """
        Initializes the RemoverModifier.

        Args:
            language (str, optional): The programming language of the code.
        """
        self.language = language
        self.config = super().DEFAULT_LANG_CONFIG.get(language, {})

        if not self.config:
            raise ValueError(f"Unsupported language: {language}")


    def modify(self, chunk, context):
        """
        Removes every word in the chunk that is not a structural token.

        Args:
            chunk (str): The feature to modify.
            context (str): The context in which the feature appears.

        Returns:
            str: The modified feature with non-structural tokens removed.
        """
        # Preserves spaces, punctuation, and dots
        line_tokens = re.split(self.config["regex_pattern"], chunk)

        # Mask only semantic tokens, reconstructing the chunk with spaces intact
        masked_line = ''.join([
            t if t in self.config["structural_tokens"]
            else ""
            for t in line_tokens
        ])

        return masked_line
