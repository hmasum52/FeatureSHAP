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
print("Printing shap values")
print(shapley_values)