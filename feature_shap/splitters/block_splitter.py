from feature_shap.splitters import SplitterBase
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


_LANGUAGE_REGISTRY = {
    'java': tsjava.language,
    'python': tspython.language,
}


class BlocksSplitter(SplitterBase):
    """
    A splitter that divides code into blocks based on its Abstract Syntax Tree (AST).

    This splitter uses `tree-sitter` to parse the code and identify logical blocks
    (e.g., method declarations, statements) as features.
    """
    def __init__(self, language: str = 'java'):
        """
        Initializes the BlocksSplitter.

        Args:
            language (str, optional): The programming language of the code.
        
        Raises:
            ValueError: If the language is not supported.
        """
        language = language.lower()
        try:
            lang_fn = _LANGUAGE_REGISTRY[language]
        except KeyError:
            supported = ', '.join(_LANGUAGE_REGISTRY)
            raise ValueError(f"Unsupported language {language}. Use one of: {supported}")

        self.language = language
        self.parser = Parser(Language(lang_fn()))


    def split(self, prompt: str):
        """
        Splits the source code into blocks based on its AST.

        Blocks are cut at:
          - the very start of the method
          - the start of each top-level statement in the method body
          - the very end of the *prompt*

        Args:
            prompt (str): The source code to split.

        Returns:
            list: A list of string-blocks that, when concatenated, reconstruct the original code.
        
        Raises:
            ValueError: If the input code cannot be parsed or no declaration is found.
        """
        try:
            data = prompt.encode('utf8')
            tree = self.parser.parse(data)
            root = tree.root_node
        except Exception as e:
            raise ValueError(f"Failed to parse the input code: {e}")

        # determine the type of declaration node based on the language
        if self.language == 'python':
            decl_type = 'function_definition'
        elif self.language == 'java':
            decl_type = 'method_declaration'
        else:
            raise ValueError("Unsupported language. Use 'java' or 'python'.")

        # find the function or the method
        decl = self._find_node_by_type(root, decl_type)
        if decl is None:
            raise ValueError(f"No {decl_type} found.")

        body = self._find_node_by_type(decl, 'block')
        statements = body.named_children if body else []

        # split at decl start, each stmt start, and end of buffer
        splits = {decl.start_byte, len(data)}
        for stmt in statements:
            splits.add(stmt.start_byte)

        boundaries = sorted(splits)

        # slice & decode
        blocks = []
        for i in range(len(boundaries)-1):
            blocks.append(data[boundaries[i]:boundaries[i+1]].decode('utf8'))

        # check that we reassembled correctly
        assert ''.join(blocks) == prompt, "Failed to completely split and reassemble the code."

        return blocks


    @staticmethod
    def _find_node_by_type(node, typ):
        """
        Recursively finds the first node in the AST whose type matches the given type.

        Args:
            node: The current node to search from.
            typ (str): The type of the node to find.

        Returns:
            Node: The found node, or None if not found.
        """
        if node.type == typ:
            return node
        for c in node.children:
            found = BlocksSplitter._find_node_by_type(c, typ)
            if found:
                return found
        return None


    def join(self, parts):
        """
        Joins the parts back into a single string.

        Args:
            parts (list): A list of string parts.

        Returns:
            str: The joined string.
        """
        return "".join(parts)
