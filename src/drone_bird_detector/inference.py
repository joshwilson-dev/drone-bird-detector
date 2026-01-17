from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from .tiling import tile_image
from .dataset import InferenceDataset
from .utils import save_coco_json, generate_cvat_labels, global_nms

def predict(model, labels, input_folder: str, device: str,
            box_nms_thresh: float = 0.2, tile_width: int = 800,
            tile_height: int = 800, overlap: int = 200,
            input_gsd: float = 0.008, batch_size: int = 4,
            output_dir: str | None = None):
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
    """
    input_folder = Path(input_folder)
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in [".jpg",".png",".tif"]]
    
    dataset = InferenceDataset(files=image_files, input_gsd=input_gsd)

    all_results = []

    for item in tqdm(dataset, desc="Processing images"):
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
                outputs = model(batch)  # List[Dict] per image in batch

                # Map predictions back to original image coordinates
                for j, output in enumerate(outputs):
                    meta = tile_meta[i+j]
                    for box, label_idx, score in zip(output["boxes"], output["labels"], output["scores"]):
                        x1, y1, x2, y2 = box.tolist()
                        # Shift to original image
                        x1 = (x1 + meta["left"]) / scale_factor
                        x2 = (x2 + meta["left"]) / scale_factor
                        y1 = (y1 + meta["top"]) / scale_factor
                        y2 = (y2 + meta["top"]) / scale_factor

                        width, height = image.size

                        all_results.append({
                            "filename": filename,
                            "label": labels[label_idx.item()],
                            "score": score.item(),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "width": int(width / scale_factor),
                            "height": int(height / scale_factor)
                        })
                del outputs, batch
        del tiles_tensor, tiles, image
        if device == "cuda":
            torch.cuda.empty_cache()
        # Global nsm
        all_results = global_nms(all_results, iou_threshold=box_nms_thresh)
    # Save predictions
    if output_dir == None: output_dir = input_folder
    
    # Save all predictions in csv
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "results.csv", index=False)

    # Save all predictions in coco format
    save_coco_json(all_results, output_dir / "coco_annotations.json")

    # Create a labels file for CVAT
    generate_cvat_labels(all_results, output_dir / "cvat_labels.json")

    print(f"Saved predictions to {Path(output_dir)}")