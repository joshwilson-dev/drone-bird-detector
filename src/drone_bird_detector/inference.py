from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from .tiling import tile_image
from .dataset import InferenceDataset
from .utils import (
    save_coco_json,
    generate_cvat_labels,
    global_nms)
from .class_filtering import (
    load_included_classes,
    build_class_mask,
    filter_class_scores,
)

def predict(model, labels, input_folder: str, device: str,
            box_nms_thresh: float = 0.2, tile_width: int = 800,
            tile_height: int = 800, overlap: int = 200,
            input_gsd: float = 0.008, batch_size: int = 4,
            output_dir: str | None = None,
            included_classes: str = None, renormalise: bool = False):
    """
    Run inference on all images in a folder.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded detection model
    labels : list of str
        Class labels
    input_folder : str
        Folder with images
    device : str
        'cpu' or 'cuda'
    tile_width, tile_height, overlap : int
        Tiling parameters
    input_gsd : float
        Actual GSD of input images
    batch_size : int
        Number of tiles per batch
    output_dir : str
        Dir to write outputs to
    box_nms_thresh : float
        non-max score iou threshold
    included_classes: str = None
        Path to txt file containing classes to include
        1 row per class. A list of all class labels is in weights/labels.txt
    renormalise: bool = False
        Whether to re-normalise scores after class removal
    """
    input_folder = Path(input_folder)
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in [".jpg",".png",".tif"]]
    
    # Load included classes and create mask
    included_classes = load_included_classes(included_classes)
    if included_classes is not None:
        class_mask = build_class_mask(labels, included_classes).to(device)
    else:
        class_mask = None
    
    dataset = InferenceDataset(files=image_files, input_gsd=input_gsd)

    all_results = []

    for item in tqdm(dataset, desc="Processing images"):
        image_results = []
        image = item["image"]
        filename = item["filename"]
        scale_factor = item["scale_factor"]

        # Tile the image
        tiles, tile_meta = tile_image(
            image,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap,
            to_tensor=True
        )

        # Batch tiles
        tiles_tensor = torch.stack(tiles)

        with torch.no_grad():
            for i in range(0, len(tiles_tensor), batch_size):
                batch = tiles_tensor[i:i+batch_size].to(device)
                outputs = model(batch)

                # Map predictions back to original image coordinates
                for j, output in enumerate(outputs):
                    meta = tile_meta[i+j]

                    boxes = output["boxes"]
                    all_fg_scores = output["all_fg_scores"]
                    bg_scores = output["bg_scores"]

                    if class_mask is not None:
                        all_fg_scores, bg_scores = filter_class_scores(
                            all_fg_scores,
                            bg_scores,
                            class_mask,
                            renormalise=renormalise,
                        )

                    for k in range(len(boxes)):
                        box = boxes[k]

                        class_scores = all_fg_scores[k]
                        background_scores = bg_scores[k]

                        score, cls_idx = class_scores.max(dim=0)
                        score = score.item()

                        # Drop detection if no included class exceeds threshold
                        if score < model.roi_heads.score_thresh:
                            continue

                        label_idx = cls_idx.item() + 1

                        class_scores = class_scores.tolist()

                        x1, y1, x2, y2 = box.tolist()

                        # Shift to original image
                        x1 = (x1 + meta["left"]) / scale_factor
                        x2 = (x2 + meta["left"]) / scale_factor
                        y1 = (y1 + meta["top"]) / scale_factor
                        y2 = (y2 + meta["top"]) / scale_factor

                        width, height = image.size

                        row = {
                            "filename": filename,
                            "label": labels[label_idx],
                            "score": score,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": int(width / scale_factor),
                            "height": int(height / scale_factor),
                            "background": float(background_scores)
                        }

                        for cls_name, cls_score in zip(labels[1:], class_scores):
                            row[f"{cls_name}"] = cls_score

                        image_results.append(row)
                del outputs, batch
        
        image_results = global_nms(
            image_results,
            iou_threshold=box_nms_thresh,
            )
        all_results.extend(image_results)

        del tiles_tensor, tiles, image
        if device == "cuda":
            torch.cuda.empty_cache()
        
    # Save predictions
    if output_dir == None:
        output_dir = input_folder
    else:
        output_dir = Path(output_dir)

    # Save all predictions in csv
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        base_cols = [
            "filename", "label", "score",
            "x1", "y1", "x2", "y2",
            "width", "height", "background"
        ]
        score_cols = [f"{cls}" for cls in labels]
        df = pd.DataFrame(columns=base_cols + score_cols)

    df.to_csv(output_dir / "results.csv", index=False)

    # Save all predictions in coco format
    save_coco_json(all_results, output_dir / "coco_annotations.json")

    # Create a labels file for CVAT
    generate_cvat_labels(all_results, output_dir / "cvat_labels.json")

    print(f"Saved predictions to {Path(output_dir)}")