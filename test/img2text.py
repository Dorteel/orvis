import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# Load the model and processor
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

# Define the prompt
prompt = "<grounding>An image of"

# Capture an image from the camera
camera = cv2.VideoCapture(0)  # Use '0' for the default camera, change if you have multiple cameras

# Allow the camera to warm up and capture a single frame
ret, cv_image = camera.read()
camera.release()  # Release the camera once done

# Check if the frame was captured successfully
if not ret:
    print("Failed to capture image from the camera.")
    exit()

# Save the image using OpenCV (optional step to mimic previous behavior)
cv2.imwrite("camera_image.jpg", cv_image)

# Reload the image from the saved file (optional, can directly use `cv_image`)
cv_image = cv2.imread("camera_image.jpg")

# Convert the OpenCV image (BGR) to RGB before sending it to the processor
cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

# Prepare the inputs using the processor
inputs = processor(text=prompt, images=cv_image_rgb, return_tensors="pt")

# Generate the output text using the model
generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=128,
)

# Decode the generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Specify `cleanup_and_extract=False` to see the raw model generation
processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
print(processed_text)

# By default, the generated text is cleaned up and entities are extracted
processed_text, entities = processor.post_process_generation(generated_text)
print(processed_text)
print(entities)

# Draw bounding boxes on the image based on the detected entities
for entity in entities:
    description, _, boxes = entity
    for box in boxes:
        # box format is [x_min, y_min, x_max, y_max] with values in range [0,1] relative to image dimensions
        x_min, y_min, x_max, y_max = box
        x_min = int(x_min * cv_image.shape[1])
        y_min = int(y_min * cv_image.shape[0])
        x_max = int(x_max * cv_image.shape[1])
        y_max = int(y_max * cv_image.shape[0])
        
        # Draw the bounding box
        cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Add the description label
        cv2.putText(cv_image, description, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Detected Objects", cv_image)
cv2.waitKey(0)  # Press any key to close the image window
cv2.destroyAllWindows()
