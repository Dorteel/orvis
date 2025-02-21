from huggingface_hub import HfApi

def get_huggingface_models(task_type: str, max_models: int = 1000):
    """
    Fetch models from Hugging Face based on a specific pipeline task type.

    Args:
        task_type (str): The pipeline_tag (e.g., 'object-detection', 'text-generation').
        max_models (int): The maximum number of models to fetch.

    Returns:
        List[Dict]: A list of model metadata dictionaries.
    """
    api = HfApi()
    all_models = []

    while len(all_models) < max_models:
        remaining = max_models - len(all_models)
        batch = list(api.list_models(filter=task_type, limit=min(100, remaining)))

        if not batch:
            break

        all_models.extend(batch)

    return all_models

# Example Usage
task = "object-detection"  # Change this to any pipeline_tag
models = get_huggingface_models(task, max_models=50)

# Print first 10 models
for model in models[:10]:
    print(model.modelId)


def get_model_size(model_name):
    api = HfApi()
    # List all models (paginated)
    models = api.list_models()
    i = 0
    # Print the first 10 model names as an example
    for model in models:
        print(model.modelId)
        i += 1
        if i==10: break

    model_info = api.model_info(model_name)
    print(f"\n{'='*50}\n{model_name}\n{'-'*50}\n")
    for key, item in vars(model_info).items():
        print(f"{key}: {item}")

    total_size = model_info.usedStorage
    print(model_info.transformersInfo.auto_model)
    print(model_info.transformersInfo.processor)
    return total_size / (1024**2)  # Convert to MB

# Example usage:
model_name = "facebook/detr-resnet-50"  # Replace with the desired model name
size_mb = get_model_size(model_name)
print(f"Model size of {model_name}: {size_mb:.2f} MB")
