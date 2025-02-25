from transformers import AutoConfig
from huggingface_hub import HfApi
import requests
import pandas as pd

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
    all_models = list(api.list_models(pipeline_tag=task_type, sort='likes'))

    return all_models


def get_model_info(model_name):
    """
    Retrieves the model metadata including auto_model, processor, config path, and id2label mapping.

    Args:
        model_name (str): The Hugging Face model identifier (e.g., "facebook/detr-resnet-50").

    Returns:
        dict or None: Model metadata dictionary if successful, None if the model is incompatible.
    """
    api = HfApi()

    try:
        # Attempt to load model configuration
        config = AutoConfig.from_pretrained(model_name)

        if not isinstance(config.to_dict(), dict):
            print(f"Skipping model '{model_name}' due to unexpected config format.")
            return None

        # Fetch model details from Hugging Face API
        try:
            model_details = api.model_info(model_name)
            size = model_details.usedStorage / (1024**2) if model_details.usedStorage else None  # Convert to MB
            architecture = config.architectures[0] if getattr(config, "architectures", None) else "Unknown"
        except requests.exceptions.HTTPError as e:
            print(f"Skipping model '{model_name}' due to API error: {e}")
            return None
        
        # Extract relevant information
        model_info = {
            "auto_model": architecture,
            "processor": "AutoImageProcessor",  # Assuming image models use this
            "config_path": config.to_dict(),  # Full config dict if available
            "size": size,  # Kept your original method
            "labels": list(config.id2label.values()) if hasattr(config, "id2label") else None
        }

        return model_info

    except ValueError as e:
        # Handle cases where the model type is unrecognized
        print(f"\t\t...skipping model '{model_name}' due to incompatibility: {str(e)}")
        return 'incompatible'

    except OSError:
        # Handle missing configuration files
        print(f"\t\t...skipping model '{model_name}' as it lacks a config.json file.")
        return 'no config'

    except TypeError as e:
        print(f"\t\t...skipping model '{model_name}' due to TypeError in config processing: {str(e)}")
        return 'incompatible'


# Example Usage
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

# task = "object-detection"  # Change this to any pipeline_tag
for task in relevant_tasks:
    models = get_huggingface_models(task)
    print(f"Processing a total of {len(models)} {task} models.")

    all_labels = set()
    useful_models = []
    incompatible_models = []
    configless_models = []
    other_faulty_models = []
    i = 1
    # Iterate over models and collect information
    for model in models:
        print(f"{i}/{len(models)}: Processing: {model.modelId}")
        model_info = get_model_info(model.modelId)
        i += 1
        if model_info == 'incompatible': incompatible_models.append(model.modelId)
        elif model_info == 'no config': configless_models.append(model.modelId)
        
        elif model_info:
            all_labels.update(model_info['labels'] or [])  # Ensure labels exist before updating
            useful_models.append(model_info)
        else:
            other_faulty_models.append(model.modelId)

    df = pd.DataFrame(useful_models)
    csv_filename = f"{task}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"{'='*50}\nSummary for {task} models:")
    print(f"Total unique labels: {len(all_labels)}")
    print(f"Total useful models: {len(useful_models)} out of {len(models)}")
    print(f"\tThere are {len(configless_models)} configless models")
    print(f"\tThere are {len(incompatible_models)} incompatible models")
    print(f"\tThere are {len(other_faulty_models)} other faulty models")
    print(f"{'='*50}\n")