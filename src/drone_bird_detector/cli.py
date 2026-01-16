import argparse
from .model import load_model
from .inference import predict

def main():
    parser = argparse.ArgumentParser(description="Drone Bird Detector CLI")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder of images to predict")
    parser.add_argument("--device", type=str, required=True, default="cpu", help="Device used to run model, cpu or cuda")
    args = parser.parse_args()

    model, labels = load_model(args.device)

    predict(model, labels, args.input_folder)

if __name__ == "__main__":
    main()