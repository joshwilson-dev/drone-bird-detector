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
    class_filter = None
    renormalise = False

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

    # Load class filter
    if class_filter != None:
        with open(class_filter) as f:
            class_filter = [line.strip() for line in f if line.strip()]

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
        output_dir=output_dir
        # class_filter = class_filter,
        # renormalise = renormalise
    )

    # Check that CSV exists and is not empty
    output_csv = output_dir / "results.csv"
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert not df.empty

    # Optional: check expected columns
    expected_columns = {"filename", "label", "score", "x1", "y1", "x2", "y2"}
    assert expected_columns.issubset(set(df.columns))

    print("Integration test passed: pipeline runs end-to-end.")