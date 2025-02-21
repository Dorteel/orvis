from huggingface_hub import HfApi
from transformers import AutoConfig

def get_model_info(model_name):
    """
    Retrieves the auto_model, processor, config file, and id2label mapping for a given model.

    Args:
        model_name (str): The Hugging Face model identifier (e.g., "facebook/detr-resnet-50").

    Returns:
        dict: A dictionary containing auto_model, processor, config path, and labels.
    """
    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)

    # Extract relevant info
    model_info = {
        "auto_model": config.architectures[0] if hasattr(config, "architectures") else "Unknown",
        "processor": "AutoImageProcessor",  # Usually, image models use AutoImageProcessor
        "config_path": config.to_dict(),  # Full config dict (could save it as a file)
        "labels": list(config.id2label.values()) if hasattr(config, "id2label") else None
    }

    return model_info

# Example usage
model_name = "facebook/detr-resnet-50"  # Replace with any object detection model
model_info = get_model_info(model_name)

# Print results
# print(f"Auto Model: {model_info['auto_model']}")
# print(f"Processor: {model_info['processor']}")
print(f"Labels: {model_info['labels']}")



def get_huggingface_task_types():
    """
    Fetch and print all available task types (pipeline tags) on Hugging Face.

    Returns:
        List[str]: A list of unique task types.
    """
    api = HfApi()
    models = list(api.list_models(limit=100000))  # Fetch a large batch to get diverse tasks
    task_types = set()  # Use a set to store unique task types

    for model in models:
        if model.pipeline_tag:  # Some models might not have a pipeline_tag
            task_types.add(model.pipeline_tag)

    return sorted(task_types)  # Return a sorted list for better readability

# Example Usage
task_types = get_huggingface_task_types()

print(f"Available Hugging Face Task Types (Total of {len(task_types)})")
for task in task_types:
    print(f"- {task}")
print()

relevant_tasks = [
    "depth-estimation",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-to-text",
    "mask-generation",
    "keypoint-detection",
    "object-detection",
    "video-classification",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    "visual-question-answering"
]

def get_models(task = 'object-detection'):
    api = HfApi()
    models = list(api.list_models(filter=task))
    return models

def get_labels(model):
    return {}

print("Summary:")
for task in relevant_tasks:
    models = get_models(task)
    for model in models:
        get_model_info()
    print(f"\t{task}: Number of models: {get_models(task)}\t Models with labels: {len(get_labels())}")

print("Total number of labels: ")