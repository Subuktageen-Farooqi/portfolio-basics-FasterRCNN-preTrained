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

# TensorFlow Hub Faster R-CNN outputs original COCO category IDs (1-90 with gaps).
# Map those sparse IDs to class names to avoid index shifts / IndexError.
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]
COCO_ID_TO_LABEL = dict(zip(COCO_CATEGORY_IDS, COCO_CLASSES))

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
    class_ids = detections['detection_classes'][0].numpy().astype(int)  # Raw sparse COCO category IDs
    scores = detections['detection_scores'][0].numpy()  # Confidence scores

    # Filter out detections below the confidence threshold
    valid_detections = (scores >= threshold) & np.isin(class_ids, COCO_CATEGORY_IDS)
    boxes = boxes[valid_detections]
    class_labels = np.array([COCO_ID_TO_LABEL[class_id] for class_id in class_ids[valid_detections]])
    scores = scores[valid_detections]

    return image_rgb, boxes, class_labels, scores

# Function to display the image with detected bounding boxes
def display_image_with_detections(image, boxes, class_labels, scores, threshold=0.5):
    height, width, _ = image.shape
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height))

            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{class_labels[i]}: {scores[i]:.2f}"
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
