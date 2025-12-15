from feature_shap.models import ModelBase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List


class HuggingFaceModel(ModelBase):
    """
    A wrapper for Hugging Face's transformer models for causal language modeling.
    """
    def __init__(self, model_name_or_path, device="auto", generation_args=None):
        """
        Initializes the HuggingFaceModel.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            device (str): The device to load the model on (e.g., 'cpu', 'cuda'). Defaults to 'auto'.
            generation_args (dict, optional): A dictionary of arguments for the
                `generate` method of the Hugging Face model. Defaults to None.
        """
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with the recommended dtype for modern LLMs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # Default generation args
        self.generation_args = generation_args or {
            "max_new_tokens": 512,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def generate(self, batch: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generates text from a given prompt or batch of prompts.

        Args:
            batch (Union[str, List[str]]): A single prompt or a list of prompts.

        Returns:
            Union[str, List[str]]: The generated text or a list of generated texts.
        """
        # Tokenize
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Inference only
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                **self.generation_args,
            )

        # Decode generated tokens
        generated_tokens = outputs[:, input_len:]
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if isinstance(batch, list):
            return generated_texts
        else:
            return generated_texts[0]
