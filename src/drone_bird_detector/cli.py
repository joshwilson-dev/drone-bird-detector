import argparse
from .model import load_model
from .inference import predict

def main():
    parser = argparse.ArgumentParser(
        description="Drone Bird Detector CLI"
        )
    
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder of images to predict"
        )
    
    parser.add_argument(
        "--input-gsd",
        type=float,
        required=True,
        help="Actual GSD of the input images in meters/pixel (all images assumed same)"
        )
    
    parser.add_argument(
        "--box-score-thresh",
        type=float,
        default=0.75,
        help="The minimum score for a detection to be kept."
        )

    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.2,
        help="The iou threshold to remove overlapping boxes (lower means less overlap is allowed)."
        )

    parser.add_argument(
        "--class-filter",
        type=str,
        default=None,
        help="Path to text file with included classes (one per line)."
        )
    
    parser.add_argument(
        "--renormalise",
        action="store_true",
        help="Renormalise probabilities after filtering classes."
        )

    parser.add_argument(
        "--tile-width",
        type=int,
        default=1200,
        help="Width of tiles in pixels"
        )
    
    parser.add_argument(
        "--tile-height",
        type=int,
        default=1200,
        help="Height of tiles in pixels"
        )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        help="Overlap between tiles in pixels"
        )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used to run model, cpu or cuda"
        )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of patches to pass to the model at once."
        )
    
    args = parser.parse_args()

    model, labels = load_model(
        box_score_thresh=args.box_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        min_size=min(args.tile_height, args.tile_width),
        max_size=max(args.tile_height, args.tile_width),
        device=args.device
        )

    # Load allowed species if given
    class_filter = None
    if args.class_filter:
        with open(args.class_filter) as f:
            class_filter = [line.strip() for line in f if line.strip()]

    predict(
        model=model,
        labels=labels,
        input_folder=args.input_folder,
        input_gsd=args.input_gsd,
        box_nms_thresh=args.box_nms_thresh,
        # box_score_thresh=args.box_score_thresh,
        # class_filter=args.class_filter,
        # renormalise=args.renormalise,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap=args.overlap,
        device=args.device,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()