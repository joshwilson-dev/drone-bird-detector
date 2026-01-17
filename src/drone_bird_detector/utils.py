import json
import random
import torch
import torchvision.ops as ops
from collections import defaultdict

def load_labels(labels_path) -> list[str]:
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_coco_json(all_results, output_json):
    """
    Convert prediction results to COCO format and save to output_json
    """
    images_dict = {}
    annotations = []
    categories_dict = {}
    annotation_id = 1
    category_id_map = {}

    for res in all_results:
        filename = res["filename"]
        label = res["label"]
        width = res["width"]
        height = res["height"]

        # Add image info
        if filename not in images_dict:

            images_dict[filename] = {
                "id": len(images_dict) + 1,
                "file_name": filename,
                "width": width,
                "height": height
            }

        image_id = images_dict[filename]["id"]

        # Add category
        if label not in category_id_map:
            cid = int(label.split(";")[0])
            category_id_map[label] = cid
            categories_dict[cid] = {"id": cid, "name": label.split(";")[-1], "supercategory": "animal"}
        cid = category_id_map[label]

        x1, y1, x2, y2 = res["x1"], res["y1"], res["x2"], res["y2"]
        bbox = [x1, y1, x2 - x1, y2 - y1]  # convert to COCO format

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cid,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "attributes": {
                "score": res.get("score", 1.0)
                }
        })
        annotation_id += 1

    coco_json = {
        "images": list(images_dict.values()),
        "annotations": annotations,
        "categories": list(categories_dict.values())
    }

    with open(output_json, "w") as f:
        json.dump(coco_json, f, indent=2)

def generate_cvat_labels(results, output_file):
    """
    Generate CVAT-compatible label JSON from inference results.

    Parameters
    ----------
    results : list of dict
        Each dict should have at least a "label" key.
    output_file : str or Path
        Path to write CVAT label JSON file.
    """
    category_set = {r["label"] for r in results}  # only detected labels
    labels_json = []

    for idx, label in enumerate(sorted(category_set), start=1):
        name = label.split(";")[-1]  # last section after the last ';'
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        labels_json.append({
            "name": name,
            "id": idx,
            "color": color,
            "type": "any",
            "attributes": [
                {
                    "name": "score",
                    "input_type": "number",
                    "mutable": True,
                    "values": ["0", "1", "0.01"],
                    "default_value": "0"
                }
            ]
        })

    with open(output_file, "w") as f:
        json.dump(labels_json, f, indent=2)

def global_nms(results, iou_threshold=0.5):
    """
    Apply global NMS across all labels for each image.

    Assumes objects are mutually exclusive in space.
    """

    filtered_results = []

    # Group by image
    results_by_image = defaultdict(list)
    for r in results:
        results_by_image[r["filename"]].append(r)

    for filename, detections in results_by_image.items():
        if len(detections) == 0:
            continue

        boxes = torch.tensor(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections],
            dtype=torch.float32
        )
        scores = torch.tensor(
            [d["score"] for d in detections],
            dtype=torch.float32
        )

        keep = ops.nms(boxes, scores, iou_threshold)

        for idx in keep.tolist():
            filtered_results.append(detections[idx])

    return filtered_results