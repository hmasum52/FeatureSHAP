<div align="center">
    <img src="img/fs_logo.svg" alt="FeatureSHAP logo" height="200"/>
</div>

--------------------------------------------------------------------------------

FeatureSHAP is a library for explaining the output of Large Language Models.
It uses the Shapley value method to assign an importance score to each feature in the input.

## Features

*   **Model-agnostic:** Can be used with any text generation model (currently supports Hugging Face, vLLM, and OpenAI models).
*   **Flexible:** Supports different ways of splitting the input into features (single token, or sequences of tokens).
*   **Extensible:** Easy to add new models, splitters, modifiers, and comparators.
*   **Batch processing:** Supports batch processing for faster analysis.

## Installation

```bash
git clone https://github.com/deviserlab/FeatureSHAP
cd FeatureSHAP
pip install -r requirements.txt
```

## Usage

Here is a simple example of how to use FeatureSHAP:

```python
from feature_shap import  (
    FeatureSHAP, HuggingFaceModel, BlocksSplitter, RemovalModifier, BertScoreComparator
)

# Initialize FeatureSHAP with the model, splitter, modifier, comparator, and custom instruction
fs = FeatureSHAP(
    model=HuggingFaceModel("Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    splitter=BlocksSplitter(language="python"),
    modifier=RemovalModifier(),
    comparator=BertScoreComparator(),
    instruction='\nWrite a single sentence summary of the code above.'
)

# Define the input
input = """
def read_files(files):
    content = ""
    for file in files:
        with open(file, 'r') as f:
            content += f.read()
    return content
""".lstrip()

# Compute the Shapley values
shapley_values, interactions = fs.analyze(input, sampling_ratio=1.0)

# Print the Shapley values
print(shapley_values)
```

## Citation
```bibtex
@article{vitale2025toward,
  title={Toward Explaining Large Language Models in Software Engineering Tasks},
  author={Vitale, Antonio and Nguyen, Khai-Nguyen and Poshyvanyk, Denys and Oliveto, Rocco and Scalabrino, Simone and Mastropaolo, Antonio},
  journal={arXiv preprint arXiv:2512.20328},
  year={2025}
}
```


## LICENSE
See `LICENSE` for details.
