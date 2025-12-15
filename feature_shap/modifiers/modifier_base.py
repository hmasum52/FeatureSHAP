from abc import ABC, abstractmethod


class ModifierBase(ABC):
    """
    Abstract base class for feature modifiers.

    Modifiers are responsible for changing a feature in some way, such as by
    masking it, removing it, or paraphrasing it.
    """
    DEFAULT_LANG_CONFIG = {
        "java": {
            "structural_tokens": {'/**', '*/', '*', '<p>', '<p/>', '{', '}', '(', ')', '[', ']', ';', '\n', '.', ','},
            "regex_pattern": r'(\s+|/\*\*|\*/|\*|<p>|<p/>|[{}()\[\];.,<>\'\"/@])'
        },
        "python": {
            "structural_tokens": {'def', 'return', 'import', 'from', '(', ')', '[', ']', '{', '}', ':', '.', ',', '\n'},
            "regex_pattern": r'(\s+|[{}()\[\];.,<>\'\"/@])'
        }
    }

    @abstractmethod
    def modify(self, sample, context):
        """
        Abstract method that subclasses must implement to modify a feature.

        Args:
            sample (str): The feature to modify.
            context (str): The context in which the feature appears.

        Returns:
            str: The modified feature.
        """
        pass
