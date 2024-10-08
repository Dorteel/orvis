import cv2
from PIL import Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Load the depth estimation model
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret will be True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the image from BGR (OpenCV default) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL image
    pil_image = Image.fromarray(frame_rgb)

    # Prepare the image for the model
    inputs = image_processor(images=pil_image, return_tensors="pt")

    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to the original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert the prediction to a format suitable for visualization
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")

    # Display the depth map
    cv2.imshow('Depth Estimation', formatted)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
