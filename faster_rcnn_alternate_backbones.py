# Task 2: Inference comparison using two other pretrained detection backbones/families.
import time
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def get_model_registry(device):
    return {
        "fasterrcnn_mobilenet_v3_large_fpn": {
            "weights": FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            "builder": fasterrcnn_mobilenet_v3_large_fpn,
            "device": device,
        },
        "retinanet_resnet50_fpn_v2": {
            "weights": RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            "builder": retinanet_resnet50_fpn_v2,
            "device": device,
        },
    }


def run_inference_for_model(model_name, config, image, score_threshold=0.5, top_k_preview=5):
    weights = config["weights"]
    model = config["builder"](weights=weights).to(config["device"])
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])

    input_tensor = preprocess(image).to(config["device"])
    start = time.perf_counter()
    with torch.inference_mode():
        output = model([input_tensor])[0]
    elapsed = time.perf_counter() - start

    keep = output["scores"] >= score_threshold
    boxes = output["boxes"][keep].detach().cpu()
    labels = output["labels"][keep].detach().cpu()
    scores = output["scores"][keep].detach().cpu()

    preview = []
    for label, score in zip(labels[:top_k_preview], scores[:top_k_preview]):
        idx = int(label.item())
        name = categories[idx] if 0 <= idx < len(categories) else f"class_{idx}"
        preview.append(f"{name}:{score.item():.2f}")

    return {
        "model_name": model_name,
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "categories": categories,
        "num_detections": int(keep.sum().item()),
        "preview": preview,
        "inference_time_s": elapsed,
    }


def save_and_display_predictions(image, result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_tensor = pil_to_tensor(image)

    box_labels = []
    for label, score in zip(result["labels"], result["scores"]):
        idx = int(label.item())
        name = result["categories"][idx] if 0 <= idx < len(result["categories"]) else f"class_{idx}"
        box_labels.append(f"{name}:{score.item():.2f}")

    drawn = draw_bounding_boxes(image_tensor, result["boxes"].to(torch.int64), labels=box_labels, width=2, colors="green")
    drawn_np = drawn.permute(1, 2, 0).numpy()

    output_path = output_dir / f"{result['model_name']}_predictions.jpg"
    Image.fromarray(drawn_np).save(output_path)

    plt.figure(figsize=(12, 8))
    plt.title(result["model_name"])
    plt.imshow(drawn_np)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compare_models_on_image(image_path, score_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = load_image(image_path)
    registry = get_model_registry(device)
    output_dir = Path("task2_outputs")

    print(f"Comparing models on: {image_path}")
    print(f"Score threshold: {score_threshold}")
    print(f"Device: {device}")

    for model_name, config in registry.items():
        result = run_inference_for_model(model_name, config, image, score_threshold=score_threshold)
        save_and_display_predictions(image, result, output_dir)

        print(f"\nModel: {result['model_name']}")
        print(f"Detections >= threshold: {result['num_detections']}")
        print(f"Top labels: {', '.join(result['preview']) if result['preview'] else 'None'}")
        print(f"Approx inference time: {result['inference_time_s']:.4f}s")


image_path = "sample_image.jpg"
score_threshold = 0.5
compare_models_on_image(image_path, score_threshold=score_threshold)
