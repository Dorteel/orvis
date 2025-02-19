import cv2
import torch
import yaml
from PIL import Image
from importlib import import_module

# Load the YAML config file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Dynamically load the processor and model classes
def load_model(config, task):
    model_info = config['models'][task]
    processor_class = getattr(import_module("transformers"), model_info['processor_class'])
    model_class = getattr(import_module("transformers"), model_info['model_class'])
    
    # Load the processor and model using Hugging Face API
    processor = processor_class.from_pretrained(model_info['name'])
    model = model_class.from_pretrained(model_info['name'])
    
    return processor, model, model_info

# Apply the task (e.g., object detection, depth estimation)
def apply_model_to_frame(task, processor, model, frame, model_info):
    # Convert the frame to RGB and to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Prepare the frame for the model
    inputs = processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if task == 'object_detection':
        # Process the results for object detection
        target_sizes = torch.tensor([pil_image.size[::-1]])  # [height, width]
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=model_info.get('threshold', 0.9))[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            # Draw bounding box
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            
            # Label detected object with confidence
            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    elif task == 'depth_estimation':
        # Interpolate the predicted depth to the original size
        predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        # Convert the depth map to a format suitable for display
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")

        # Display the depth map on the frame
        depth_map = cv2.applyColorMap(formatted, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.6, depth_map, 0.4, 0)

    return frame

def main():
    # Load configuration from YAML file
    config_path = 'test/config.yaml'
    config = load_config(config_path)

    # Select the task you want to perform ('object_detection', 'depth_estimation', etc.)
    task = 'object_detection'  # Example: Choose the task dynamically

    # Load the model and processor for the task
    processor, model, model_info = load_model(config, task)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera

    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        # Capture frame-by-frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Apply the task model to the frame
        frame = apply_model_to_frame(task, processor, model, frame, model_info)

        # Display the frame with results
        cv2.imshow(f'{task} - Camera Feed', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
