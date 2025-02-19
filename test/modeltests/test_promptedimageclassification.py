import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def process_images(directory_path, text_prompts):
    """
    Process all .jpg images in the given directory with CLIP model.
    
    :param directory_path: Path to the directory containing images.
    :param text_prompts: List of text prompts to compare against images.
    """
    print(f"Checking directory: {directory_path}")
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".png")]
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
        inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        
        # Run the model
        with torch.no_grad():
            outputs = model(**inputs)
        
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

# Example usage
if __name__ == "__main__":
    directory = "/home/user/pel_ws/src/orvis/obs_graphs/run_20250205160423"  # Updated path
    prompts = ["fruit", "apple", "banana"]  # Modify as needed
    results = process_images(directory, prompts)
    
    print("\nFinal Output:")
    for image_name, scores in results.items():
        print(f"Results for {image_name}:")
        for text, prob in scores.items():
            print(f"  {text}: {prob:.4f}")
