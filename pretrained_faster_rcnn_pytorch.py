# Task 1: PyTorch implementation of the Faster R-CNN tutorial workflow.
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def load_model(device):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model = model.to(device)
    model.eval()
    categories = weights.meta.get("categories", [])
    preprocess = weights.transforms()
    return model, preprocess, categories


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def detect_objects_pytorch(image, model, preprocess, device, score_threshold=0.5):
    input_tensor = preprocess(image).to(device)
    with torch.inference_mode():
        output = model([input_tensor])[0]

    scores = output["scores"]
    keep = scores >= score_threshold
    filtered = {
        "boxes": output["boxes"][keep].cpu(),
        "labels": output["labels"][keep].cpu(),
        "scores": output["scores"][keep].cpu(),
    }
    return filtered


def draw_detections(image, detections, categories):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(detections["boxes"], detections["labels"], detections["scores"]):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        label_idx = int(label.item())
        class_name = categories[label_idx] if 0 <= label_idx < len(categories) else f"class_{label_idx}"
        ax.text(x1, max(0, y1 - 5), f"{class_name}: {score.item():.2f}", color="lime", fontsize=9, backgroundcolor="black")

    ax.axis("off")
    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "sample_image.jpg"
score_threshold = 0.5

model, preprocess, categories = load_model(device)
image = load_image(image_path)
detections = detect_objects_pytorch(image, model, preprocess, device, score_threshold=score_threshold)
draw_detections(image, detections, categories)
