from feature_shap.splitters import SplitterBase
import os
import json
from typing import List
from openai import OpenAI

class LLMSplitter(SplitterBase):
    """
    A splitter that uses a language model (LLM) to segment a docstring into granular sections.

    This splitter leverages the OpenAI API to intelligently break down a docstring
    into meaningful coding sentence segments, preserving original formatting.
    """
    def __init__(self, api_key=None, model="gpt-4.1-nano"):
        """
        Initializes the LLMSplitter.

        Args:
            model (str, optional): The OpenAI model to use for splitting.
        """
        self.model = model
        try:
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")

            self.client = OpenAI(api_key=api_key)
        except Exception:
            raise ValueError("Failed to initialize OpenAI client. Please check your API key and environment variables.")
        self.system_prompt = (
            "You are a docstring segmenter. "
            "Split the user given docstring into granular sections. \n"
            "CRITICAL REQUIREMENTS:\n"
            "- You MUST always output a list of strings - no other format is acceptable\n"
            "- You MUST NOT paraphrase, rewrite, or modify any text - only extract exactly as written\n"
            "- Each string must be an exact substring from the original docstring\n"
            "- Include all original whitespace, newlines, and formatting exactly as they appear\n"
            "- When concatenated with ''.join(list), it must perfectly reconstruct the original docstring\n"

            "JSON FORMATTING REQUIREMENTS:\n"
            "- Output ONLY valid JSON - an object with numeric keys and string values: {\"0\": \"string1\", \"1\": \"string2\", \"2\": \"string3\"}\n"
            "- Each string value must be a meaningful coding sentence segments.\n"
            "- Every key must be a string representing the index: \"0\", \"1\", \"2\", etc.\n"
            "- Every value must be a string segment enclosed in double quotes\n"
            "- Ensure all quotes and special characters are properly escaped\n"
            
            "Here are examples, your response should follow this:\n\n"
            "Input: \"Search for a specific string within the JSON data of files in a given directory and its subdirectories. This function recursively scans the specified directory for JSON files, then checks each file to see if the given string is present within the JSON data structure.\\nNote that: The string search is case-sensitive and looks for a match within the structure of the JSON data, not just as a substring in the file content. If the directory does not contain any JSON files or if no JSON files contain the string, an empty list is returned.\\nThe function should output with:\\n    list: A list of file paths (str) containing the string within their JSON data.\\nYou should write self-contained code starting with:\\n```\\nimport json\\nfrom glob import glob\\ndef task_func(directory, string):\\n```\"\n"
            "Output: {\"0\": \"Search for a specific string within the JSON data of files in a given directory and its subdirectories. \", \"1\": \"This function recursively scans the specified directory for JSON files, \", \"2\": \"then checks each file to see if the given string is present within the JSON data structure.\\n\", \"3\": \"Note that: \", \"4\": \"The string search is case-sensitive and looks for a match within the structure of the JSON data,\", \"5\": \" not just as a substring in the file content. \", \"6\": \"If the directory does not contain any JSON files or if no JSON files contain the string, an empty list is returned.\\n\", \"7\": \"The function should output with:\\n\", \"8\": \"    list: A list of file paths (str) containing the string within their JSON data.\\n\", \"9\": \"You should write self-contained code starting with:\\n```\\nimport json\\nfrom glob import glob\\ndef task_func(directory, string):\\n```\"}\n\n"
            "Input: \"Calculate the median of all elements in a nested list 'L'.\\nThe function should output with:\\n    median (float): The median.\\nYou should write self-contained code starting with:\\n```\\nimport numpy as np\\nimport math\\ndef task_func(L):\\n```\"\n"
            "Output: {\"0\": \"Calculate the median of all elements in a nested list 'L'.\\n\", \"1\": \"The function should output with:\\n    \", \"2\": \"median (float): The median.\\n\", \"3\": \"You should write self-contained code starting with:\\n```\\nimport numpy as np\\nimport math\\ndef task_func(L):\\n```\"}\n\n"
            "Input: \"Count the frequency of each letter in a list of lists. If a list is empty, fill it with a random sample from the alphabet, and then count the letters.\\nThe function should output with:\\n    Counter: A Counter object with the frequency of each letter.\\nYou should write self-contained code starting with:\\n```\\nfrom collections import Counter\\nimport itertools\\nimport random\\n# Constants\\nALPHABET = 'abcdefghijklmnopqrstuvwxyz'\\ndef task_func(list_of_lists, seed=0):\\n```\"\n"
            "Output: {\"0\": \"Count the frequency of each letter in a list of lists. \", \"1\": \"If a list is empty, fill it with a random sample from the alphabet, \", \"2\": \"and then count the letters.\\n\", \"3\": \"The function should output with:\\n    \", \"4\": \"Counter: A Counter object with the frequency of each letter.\\n\", \"5\": \"You should write self-contained code starting with:\\n```\\nfrom collections import Counter\\nimport itertools\\nimport random\\n# Constants\\nALPHABET = 'abcdefghijklmnopqrstuvwxyz'\\ndef task_func(list_of_lists, seed=0):\\n```\"}\n\n"
            "Input: \"Generate a Pandas DataFrame with random numeric values between 1 and 100, inclusive, and replace all occurrences of values less than 10 with -1.\\nThe function should output with:\\n    DataFrame: The modified Pandas DataFrame.\\nYou should write self-contained code starting with:\\n```\\nimport pandas as pd\\nimport numpy as np\\ndef task_func(data_size=1000, column_names=['A', 'B', 'C', 'D', 'E'], seed=0):\\n```\"\n"
            "Output: {\"0\": \"Generate a Pandas DataFrame\", \"1\": \" with random numeric values between 1 and 100, \", \"2\": \"inclusive, \", \"3\": \"and replace all occurrences of values less than 10 with -1.\\n\", \"4\": \"The function should output with:\\n    \", \"5\": \"DataFrame: The modified Pandas DataFrame.\\n\", \"6\": \"You should write self-contained code starting with:\\n```\\nimport pandas as pd\\nimport numpy as np\\ndef task_func(data_size=1000, column_names=['A', 'B', 'C', 'D', 'E'], seed=0):\\n```\"}\n\n"
            "Input: \"Save a Pandas DataFrame to a JSON file in a specified directory.\\nNote that: The function manipulates a Pandas DataFrame and saves it as a JSON file.\\nThe function should output with:\\n    str: The full file path where the DataFrame is saved.\\nYou should write self-contained code starting with:\\n```\\nimport pandas as pd\\nimport os\\nOUTPUT_DIR = './output'\\ndef task_func(df, filename, output_dir=OUTPUT_DIR):\\n```\"\n"
            "Output: {\"0\": \"Save a Pandas DataFrame \", \"1\": \"to a JSON file \", \"2\": \"in a specified directory.\\n\", \"3\": \"Note that: \", \"4\": \"The function manipulates a Pandas DataFrame \", \"5\": \"and saves it as a JSON file.\\n\", \"6\": \"The function should output with:\\n    \", \"7\": \"str: The full file path where the DataFrame is saved.\", \"8\": \"\\nYou should write self-contained code starting with:\\n```\\nimport pandas as pd\\nimport os\\nOUTPUT_DIR = './output'\\ndef task_func(df, filename, output_dir=OUTPUT_DIR):\\n```\"}\n\n"
        )


    def _parse_docstring_segments(
        self,
        docstring: str,
        model: str = "gpt-4.1",
        temperature: float = 0.0
    ) -> List[str]:
        """
        Split a Python docstring into functional segments via the OpenAI API.
    
        Args:
            docstring (str): The full raw docstring.
            model (str): The OpenAI model to use.
            temperature (float): Sampling temperature; 0 for deterministic.
    
        Returns:
            List[str]: A list of strings, each one a contiguous slice of the original docstring
            corresponding to a single section (short desc, long desc, Args, Returns, Raises, Examples).
            When concatenated with ''.join(...), they exactly reconstruct `docstring`.
        """
        user_prompt = f'Here is the docstring to segment:\n"""\n{docstring}\n"""\n\nSplit it.'

        resp = self.client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        segments = resp.output_text

        segments_dict = json.loads(segments)
        parts = [segments_dict[str(i)] for i in range(len(segments_dict))]

        return parts
    
    def split(self, prompt):
        """
        Splits a prompt (docstring) into segments using the LLM.

        Args:
            prompt (str): The docstring to split.

        Returns:
            List[str]: A list of docstring segments.
        """
        return self._parse_docstring_segments(prompt)
        
    def join(self, parts):
        """
        Joins a list of docstring segments back into a single string.

        Args:
            parts (List[str]): A list of docstring segments.

        Returns:
            str: The joined docstring.
        """
        return "".join(parts).strip()
