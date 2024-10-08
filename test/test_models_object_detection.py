import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

# Load the processor and model for object detection
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    
    # If frame is read correctly, ret will be True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the image from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(frame_rgb)

    # Prepare the image for the model
    inputs = processor(images=pil_image, return_tensors="pt")

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs to bounding boxes and class logits, filter by score threshold
    target_sizes = torch.tensor([pil_image.size[::-1]])  # [height, width]
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw the bounding boxes and labels on the frame
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

        # Draw bounding box
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        # Label detected object with confidence
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the bounding boxes
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
