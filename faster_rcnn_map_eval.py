# Task 3: Dataset-level mAP evaluation for a pretrained object detector (COCO-format).
from pathlib import Path
import json

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except Exception:
    MeanAveragePrecision = None


DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_FILE = DATASET_ROOT / "annotations" / "instances.json"
BATCH_SIZE = 1
SCORE_THRESHOLD = 0.05


class COCODetectionDataset(Dataset):
    def __init__(self, images_dir, coco, transforms, category_id_map):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.category_id_map = category_id_map

        self.images = sorted(coco["images"], key=lambda x: x["id"])

        ann_by_image = {}
        for ann in coco["annotations"]:
            ann_by_image.setdefault(ann["image_id"], []).append(ann)
        self.ann_by_image = ann_by_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        anns = self.ann_by_image.get(image_info["id"], [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            raw_category_id = ann["category_id"]
            if raw_category_id not in self.category_id_map:
                raise KeyError(f"Unknown category_id {raw_category_id} in annotation for image_id={image_info['id']}")
            labels.append(self.category_id_map[raw_category_id])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([image_info["id"]], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        image_tensor = self.transforms(image)
        return image_tensor, target


def build_category_id_map(coco_categories, model_categories):
    dataset_id_to_name = {cat["id"]: cat["name"].strip().lower() for cat in coco_categories if "id" in cat and "name" in cat}

    model_name_to_label = {
        name.strip().lower(): idx
        for idx, name in enumerate(model_categories)
        if name and name != "N/A"
    }

    if dataset_id_to_name and model_name_to_label:
        missing_names = sorted({name for name in dataset_id_to_name.values() if name not in model_name_to_label})
        if missing_names:
            missing_preview = ", ".join(missing_names[:5])
            raise ValueError(
                "Dataset category names are not aligned with model labels. "
                f"Missing matches for: {missing_preview}"
            )

        return {dataset_id: model_name_to_label[name] for dataset_id, name in dataset_id_to_name.items()}

    dataset_ids = {cat["id"] for cat in coco_categories if "id" in cat}
    model_valid_ids = {
        idx for idx, name in enumerate(model_categories) if name and name != "N/A"
    }

    if dataset_ids and dataset_ids.issubset(model_valid_ids):
        return {dataset_id: dataset_id for dataset_id in dataset_ids}

    raise ValueError(
        "Unable to build category-id mapping from dataset categories to model label space. "
        "Provide COCO categories with names that match the model's category names."
    )


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def evaluate_map(dataset_root=DATASET_ROOT, score_threshold=SCORE_THRESHOLD):
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    annotations_file = dataset_root / "annotations" / "instances.json"

    if MeanAveragePrecision is None:
        print("torchmetrics is not available. Install torchmetrics to compute mAP/mAP50/mAP75.")
        return

    if not images_dir.exists() or not annotations_file.exists():
        raise FileNotFoundError(
            "Expected COCO-format dataset at dataset/images and dataset/annotations/instances.json"
        )

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    category_id_map = build_category_id_map(
        coco_categories=coco.get("categories", []),
        model_categories=weights.meta.get("categories", []),
    )

    dataset = COCODetectionDataset(
        images_dir=images_dir,
        coco=coco,
        transforms=weights.transforms(),
        category_id_map=category_id_map,
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.inference_mode():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            for out in outputs:
                keep = out["scores"] >= score_threshold
                preds.append(
                    {
                        "boxes": out["boxes"][keep].detach().cpu(),
                        "scores": out["scores"][keep].detach().cpu(),
                        "labels": out["labels"][keep].detach().cpu(),
                    }
                )

            target_list = []
            for tgt in targets:
                target_list.append(
                    {
                        "boxes": tgt["boxes"].detach().cpu(),
                        "labels": tgt["labels"].detach().cpu(),
                    }
                )

            metric.update(preds, target_list)

    results = metric.compute()
    model_name = "fasterrcnn_resnet50_fpn"
    split_name = "dataset/images + dataset/annotations/instances.json"

    print(f"Model: {model_name}")
    print(f"Category ID mapping entries: {len(category_id_map)}")
    print(f"Evaluated split: {split_name}")
    print(f"mAP: {results['map'].item():.4f}")
    print(f"mAP50: {results['map_50'].item():.4f}")
    if "map_75" in results:
        print(f"mAP75: {results['map_75'].item():.4f}")


evaluate_map()
