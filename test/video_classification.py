from transformers import AutoImageProcessor, TimesformerForVideoClassification
import cv2
import numpy as np
import torch

# Initialize the processor and model
processor = AutoImageProcessor.from_pretrained("facebook/timesformer-hr-finetuned-k600")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-hr-finetuned-k600")

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Set the number of frames required by the model (e.g., 16)
num_frames = 16
frame_height, frame_width = 448, 448  # Dimensions expected by the model
video_frames = []

# Capture frames from the webcam
while len(video_frames) < num_frames:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    
    # Resize frame to match the model input size
    resized_frame = cv2.resize(frame, (frame_width, frame_height))
    
    # Convert BGR (OpenCV format) to RGB (model expects RGB)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Append the frame to video_frames list
    video_frames.append(rgb_frame)
    
    # Display the frame (optional)
    cv2.imshow("Webcam Feed", frame)
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Ensure we have the correct number of frames by duplicating or truncating
if len(video_frames) < num_frames:
    print(f"Insufficient frames captured. Only {len(video_frames)} frames available.")
    exit()
elif len(video_frames) > num_frames:
    video_frames = video_frames[:num_frames]

# Pre-process video frames for the model
inputs = processor(images=video_frames, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted class label
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
