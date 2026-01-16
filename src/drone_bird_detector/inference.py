import torch
from pathlib import Path
import pandas as pd
from PIL import Image


def predict_image(model, labels, image_path: str):
    """Run inference on a single image and return predictions."""
    image = Image.open(image_path).convert("RGB")
    # transforms
    predictions = model(image)
    # map predictions to labels
    return {"image": image_path.name, "prediction": "example_label"}

def predict(model, labels, folder_path: str, output_csv: str = "results.csv"):
    """Run inference over all images in a folder and save to CSV."""
    folder = Path(folder_path)
    results = []
    for img_file in folder.glob("*.jpg"):
        pred = predict_image(model, labels, img_file)
        results.append(pred)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")