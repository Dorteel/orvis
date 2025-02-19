import os
import torch
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from orvis.srv import PromptedImageClassificationResponse

class PromptedImageClassifier:
    def __init__(self, config):
        self.config = config

        # Dynamically import model and processor classes
        model_class_path = config['imports']['model_class']
        self.model_class = self.dynamic_import(model_class_path)
        
        processor_class_path = config['imports']['processor_class']
        self.processor_class = self.dynamic_import(processor_class_path)

        # Load the model and processor
        self.model = self.model_class.from_pretrained(config['model']['model_name'])
        self.processor = self.processor_class.from_pretrained(config['processor']['processor_name'])
    
    def process_images(self, directory_path, text_prompts):
        """
        Process all .jpg images in the given directory with CLIP model.
        
        :param directory_path: Path to the directory containing images.
        :param text_prompts: List of text prompts to compare against images.
        """
        print(f"Checking directory: {directory_path}")
        image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".jpg")]
        print(f"Found {len(image_files)} images: {image_files}")
        
        results = {}
        
        for image_file in image_files:
            image_path = os.path.join(directory_path, image_file)
            print(f"Processing image: {image_file}")
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue
            
            print(f"Running model on {image_file} with prompts: {text_prompts}")
            inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
            
            # Run the model
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Compute similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).tolist()[0]
            
            results[image_file] = {text: prob for text, prob in zip(text_prompts, probs)}
            
            # Display image with probabilities
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis("off")
            title_text = "\n".join([f"{text}: {prob:.4f}" for text, prob in zip(text_prompts, probs)])
            ax.set_title(title_text, fontsize=12)
            plt.show()
        
        return results
    
    def dynamic_import(self, import_path):
        """
        Dynamically import the class from the import path string.
        """
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

# Example usage
if __name__ == "__main__":
    config = {
        "imports": {
            "model_class": "transformers.CLIPModel",
            "processor_class": "transformers.CLIPProcessor"
        },
        "model": {
            "model_name": "openai/clip-vit-large-patch14"
        },
        "processor": {
            "processor_name": "openai/clip-vit-large-patch14"
        }
    }
    classifier = PromptedImageClassifier(config)
    directory = "/home/user/pel_ws/src/orvis/obs_graphs/run_20250205160423"  # Updated path
    prompts = ["a photo of a cat", "a photo of a dog"]  # Modify as needed
    results = classifier.process_images(directory, prompts)
    
    print("\nFinal Output:")
    for image_name, scores in results.items():
        print(f"Results for {image_name}:")
        for text, prob in scores.items():
            print(f"  {text}: {prob:.4f}")
