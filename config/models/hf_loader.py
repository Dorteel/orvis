import yaml
import json
from huggingface_hub import hf_hub_download
import transformers

def get_model_info(model_name):
    """Fetch model configuration from Hugging Face Hub."""
    try:
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error retrieving model config: {e}")
        return None

def generate_import_mappings():
    """Dynamically generate import mappings for Hugging Face models."""
    import_mappings = {}
    model_mapping = transformers.models.auto.modeling_auto.MODEL_MAPPING_NAMES
    
    for model_class, class_name in model_mapping.items():
        if "ImageClassification" in class_name:
            import_mappings[class_name] = ("transformers.DetrForImageClassification", "transformers.DetrImageProcessor")
        elif "ObjectDetection" in class_name:
            import_mappings[class_name] = ("transformers.DetrForObjectDetection", "transformers.DetrImageProcessor")
        elif "SequenceClassification" in class_name:
            import_mappings[class_name] = ("transformers.AutoModelForSequenceClassification", "transformers.AutoTokenizer")
        else:
            import_mappings[class_name] = ("transformers.AutoModel", "transformers.AutoProcessor")
    return import_mappings

def generate_yaml(model_name):
    """Generate a YAML file with model information."""
    config = get_model_info(model_name)
    if not config:
        return
    
    architecture = config.get("architectures", ["AutoModel"])[0]
    import_mappings = generate_import_mappings()
    model_class, processor_class = import_mappings.get(architecture, ("transformers.AutoModel", "transformers.AutoProcessor"))
    labels = list(config.get("id2label").values())
    yaml_data = {
        "annotator": {
            "name": model_name.replace("/", "_"),
            "type": architecture,
            "task_type": "ImageClassification" if "Classification" in architecture else "ObjectDetection",
            "detected_property": "Label" if "Classification" in architecture else "ObjectType",
        },
        "imports": {
            "model_class": model_class,
            "processor_class": processor_class,
        },
        "model": {
            "model_name": model_name,
        },
        "processor": {
            "processor_name": model_name,
        },
        "ros": {
            "result_topic": f"annotators/{model_name.replace('/', '_')}/result",
            "image_topic": f"annotators/{model_name.replace('/', '_')}/image",
        },
        "detection": {
            "labels": labels,
            "confidence_threshold": 0.9,
            "iou_threshold": 0.45,
        },
    }
    
    yaml_filename = f"{model_name.replace('/', '_')}.yaml"
    with open(yaml_filename, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    
    print(f"YAML configuration saved as {yaml_filename}")

# Example usage
generate_yaml("ioanasong/vit-MINC-2500")
