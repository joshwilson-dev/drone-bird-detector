import torch
from PIL import Image
from pathlib import Path

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, files, input_gsd: float):
        """
        files : list of image file paths
        input_gsd : actual GSD of the images in meters/pixel
        """
        self.files = [Path(f) for f in files if f.suffix.lower() in [".jpg", ".png", ".tif"]]
        self.input_gsd = input_gsd
        self.target_gsd = 0.005

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = path.name

        image = Image.open(path).convert("RGB")
        width, height = image.size
        scale_factor = self.input_gsd / self.target_gsd
        if scale_factor != 1.0:
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, resample=Image.BILINEAR)

        return {
            "image": image,
            "filename": filename,
            "scale_factor": scale_factor,
            "width": width,
            "height": height
        }

    def __len__(self):
        return len(self.files)