from feature_shap.modifiers.modifier_base import ModifierBase
import re


class MaskerModifier(ModifierBase):
    """
    A modifier that replaces non-structural tokens in a feature with a mask token.

    This is useful for understanding the importance of the semantic content of a feature,
    as opposed to its structural role.
    """
    def __init__(self, mask_token="|MASKED|", language="java"):
        """
        Initializes the MaskerModifier.

        Args:
            mask_token (str, optional): The token to use for masking.
            language (str, optional): The programming language of the code.
        """
        self.mask_token = mask_token
        self.language = language
        self.config = super().DEFAULT_LANG_CONFIG.get(language, {})

        if not self.config:
            raise ValueError(f"Unsupported language: {language}")


    def modify(self, chunk, context):
        """
        Replaces every word in the chunk that is not a structural token with the mask token.

        Args:
            chunk (str): The feature to modify.
            context (str): The context in which the feature appears.

        Returns:
            str: The modified feature with non-structural tokens masked.
        """
        # Preserves spaces, punctuation, and dots
        line_tokens = re.split(self.config["regex_pattern"], chunk)

        # Mask only semantic tokens, reconstructing the chunk with spaces intact
        masked_line = ''.join([
            t if t in self.config["structural_tokens"] or t.strip() in "{}();[],.<>\n" or t.isspace()
            else self.mask_token
            for t in line_tokens
        ])

        # Ensure newlines are untouched in the final line
        masked_line = masked_line.replace(f"{self.mask_token}\n", "\n")

        return masked_line
