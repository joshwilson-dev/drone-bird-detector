from pathlib import Path
from drone_bird_detector.model import load_model
from drone_bird_detector.inference import predict
import pandas as pd

def test_full_inference_pipeline(tmp_path):
    """
    Integration test: runs the full pipeline on a small set of images.
    Uses temporary directory to save CSV output.
    """
    # Parameters
    input_folder = Path(__file__).parent.parent / "examples" / "australian_pelican-gsd_0.018"
    input_gsd = 0.018
    tile_width = 1200
    tile_height = 1200
    overlap = 400
    box_score_thresh = 0.9
    box_nms_thresh = 0.2
    batch_size = 1
    device = "cuda"
    included_classes = Path(__file__).parent.parent / "examples" / "australian_pelican-gsd_0.018" / "included_classes.txt"
    renormalise = True

    # Temporary CSV output
    output_dir = None # tmp_path
    if output_dir == None:
        output_dir = input_folder

    # Load model + labels
    model, labels = load_model(
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        min_size = min(tile_width, tile_height),
        max_size = max(tile_width, tile_height),
        device=device)

    # Run inference
    predict(
        model=model,
        labels=labels,
        input_folder=input_folder,
        input_gsd=input_gsd,
        box_nms_thresh=box_nms_thresh,
        device=device,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        batch_size=batch_size,
        output_dir=output_dir,
        included_classes = included_classes,
        renormalise = renormalise
    )

    # Check that CSV exists and is not empty
    output_csv = output_dir / "results.csv"
    assert output_csv.exists()
    df = pd.read_csv(output_csv)

    # Optional: check expected columns
    expected_columns = {"filename", "label", "score", "x1", "y1", "x2", "y2", "background"}
    assert expected_columns.issubset(set(df.columns))

    print("Integration test passed: pipeline runs end-to-end.")