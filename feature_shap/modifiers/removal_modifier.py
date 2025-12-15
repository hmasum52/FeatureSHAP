from feature_shap.modifiers.modifier_base import ModifierBase

class RemovalModifier(ModifierBase):
    """
    A modifier that removes a feature entirely.

    This is the simplest form of perturbation, where the feature is simply
    replaced with an empty string.
    """
    def __init__(self):
        """
        Initializes the RemovalModifier.
        """
        pass

    def modify(self, chunk, context):
        """
        Removes the feature entirely.

        Args:
            chunk (str): The feature to remove.
            context (str): The context in which the feature appears.

        Returns:
            str: An empty string.
        """
        return ""
