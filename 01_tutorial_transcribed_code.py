import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

# Step 1: Load the Pre-trained Faster R-CNN Model from TensorFlow Hub
MODEL_URL = "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"
model = hub.load(MODEL_URL)

# Class labels for COCO Dataset (used by the pre-trained model)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to perform object detection
def detect_objects(image_path, model, threshold=0.5):
    # Debug message to verify the file path
    print(f"Loading image from: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Please check the file path.")

    # Convert to RGB format for visualization and TensorFlow processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to the model's expected input size
    input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run object detection
    detections = model(input_tensor)

    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()  # Bounding boxes
    class_indices = detections['detection_classes'][0].numpy().astype(int) - 1  # Class indices, -1 to adjust for zero-indexed
    scores = detections['detection_scores'][0].numpy()  # Confidence scores

    # Filter out detections below the confidence threshold
    valid_detections = scores >= threshold
    boxes = boxes[valid_detections]
    class_indices = class_indices[valid_detections]
    scores = scores[valid_detections]

    return image_rgb, boxes, class_indices, scores

# Function to display the image with detected bounding boxes
def display_image_with_detections(image, boxes, class_indices, scores, threshold=0.5):
    height, width, _ = image.shape
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))

            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{COCO_CLASSES[class_indices[i]]}: {scores[i]:.2f}"
            cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the result with matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage with your image path
image_path = "D:\\Deep Learning Course\\Codes\\Images for pretrained\\Sample image.jpg"
try:
    image_rgb, detected_boxes, detected_classes, detected_scores = detect_objects(image_path, model, threshold=0.5)
    display_image_with_detections(image_rgb, detected_boxes, detected_classes, detected_scores)
except FileNotFoundError as e:
    print(e)
