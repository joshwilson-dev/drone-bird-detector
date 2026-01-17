from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .utils import load_labels

MODEL_DIR = Path(__file__).parent.parent / "weights"
HF_REPO_ID = "JoshuaWilson/drone-bird-detector"
HF_MODEL_FILE = "model.pth"
HF_LABEL_FILE = "labels.txt"
HF_REVISION = "main"

kwargs = {
    "rpn_pre_nms_top_n_test": 250,
    "rpn_post_nms_top_n_test": 250,
    "rpn_nms_thresh": 0.5,
    "rpn_score_thresh": 0.01,
    "box_detections_per_img": 100,
    "min_size": 800,
    "max_size": 800}

def download_model_if_needed() -> Path:
    """Download model from Hugging Face if it doesn't exist locally."""
    local_model_path = MODEL_DIR / "model.pth"
    if not local_model_path.exists():
        print("Model not found locally. Downloading from Hugging Face...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILE,
            revision=HF_REVISION,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to {local_model_path}")
    return local_model_path

def download_labels_if_needed() -> Path:
    """Download labels.txt from Hugging Face if missing."""
    local_label_path = MODEL_DIR / HF_LABEL_FILE
    if not local_label_path.exists():
        print("Labels file not found locally. Downloading from Hugging Face...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_LABEL_FILE,
            revision=HF_REVISION,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Labels downloaded to {local_label_path}")
    return local_label_path

def load_model(box_score_thresh, box_nms_thresh, min_size, max_size, device):
    """Load the PyTorch model, downloading if necessary."""

    # update the model parameters with args
    kwargs["min_size"] = min_size
    kwargs["max_size"] = max_size
    kwargs["box_nms_thresh"] = box_nms_thresh
    kwargs["box_score_thresh"] = box_score_thresh

    # check if model exists, if not download it.
    model_file = download_model_if_needed()

    # check if labels exists, if not download it.
    labels_path = download_labels_if_needed()
    labels = load_labels(labels_path)

    # create the model and load weights
    backbone = resnet_fpn_backbone(
        backbone_name = "resnet101",
        weights = None
        )
    
    model = torchvision.models.detection.__dict__["FasterRCNN"](
        backbone = backbone,
        num_classes = len(labels),
        **kwargs
        )
    
    checkpoint = torch.load(model_file, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    # put model in inference mode and move to target device
    torch.backends.cudnn.deterministic = True
    model.eval()
    model.to(device)

    return model, labels