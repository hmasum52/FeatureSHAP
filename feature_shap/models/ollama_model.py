from feature_shap.models.model_base import ModelBase
from typing import Union, List
from openai import OpenAI


class OllamaModel(ModelBase):
    """
    A wrapper for Ollama models that conforms to ModelBase.
    Uses OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434/v1",
        generation_args: dict | None = None,
    ):
        """
        Initializes the OllamaModel.

        Args:
            model_name (str, optional): The name of the Ollama model to use (e.g., "llama2", "mistral").
            base_url (str, optional): The Ollama server base URL. Defaults to http://localhost:11434/v1.
            generation_args (dict | None, optional): A dictionary of arguments for the Ollama API.
        """
        self.model_name = model_name
        self.base_url = base_url

        try:
            # Initialize OpenAI client pointing to Ollama server
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"  # Ollama doesn't require a real API key
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Ollama client: {str(e)}")

        # Default generation args
        self.generation_args = generation_args or {
           # "temperature": 0.0,
            "max_tokens": 512,
        }


    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate text from a prompt or batch of prompts using Ollama.

        Args:
            batch (Union[str, List[str]]): A single prompt or list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or list of generated texts.
        """
        prompts = [batch] if isinstance(batch, str) else batch

        outputs = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.generation_args,
            )

            # Extract generated text
            outputs.append(response.choices[0].message.content.strip())

        return outputs if isinstance(batch, list) else outputs[0]
