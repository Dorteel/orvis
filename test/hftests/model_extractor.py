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
