# Drone Bird Detector

A command-line tool for detecting and identifying birds in drone imagery for ecological surveys.
This software accompanies:

**Wilson, J. P., Amano, T., Bregnballe, T., Corregidor-Castro, A., Francis, R., Gallego-García, D., Hodgson, J. C., Jones, L. R., Luque-Fernández, C. R., Marchowski, D., McEvoy, J., McKellar, A. E., Oosthuizen, W. C., Pfeifer, C., Renner, M., Sarasola, J. H., Sokač, M., Valle, R., Zbyryt, A., & Fuller, R. A. (2026). Big Bird: A global dataset of birds in drone imagery annotated to species level. Remote Sensing in Ecology and Conservation. https://doi.org/10.1002/rse2.70059**

---

## Features

* Detects birds in aerial drone images using a deep learning model.
* Identifies species, age category, and sex for each detected bird.
* Outputs results in a CSV file suitable for ecological analysis.
* Lightweight CLI tool: select a folder of images and run inference.
* Supports CPU and GPU execution (if PyTorch + CUDA is installed).

---

## Limitations

Model performance is context-dependent and may not generalise across sites, species, or sensors.
A proportion of detections should always be manually reviewed.
Manual review is necessary to quantify model recall for each dataset context.

---

## Installation

The package can be installed from GitHub. GPU support is optional and can be enabled after installation.

### From GitHub (latest development version)

```bash
git clone https://github.com/joshwilson-dev/drone-bird-detector.git
cd drone-bird-detector
python -m venv .venv          # optional but recommended
source .venv/bin/activate     # Linux/macOS
source .venv/Scripts/activate # Windows
pip install -e .
```

### GPU support (optional)

By default, `drone-bird-detector` installs a **CPU-only** version of PyTorch, which works on any machine. 

However, the CPU version is likely to run very slowly.

If you have an NVIDIA GPU and want to enable GPU acceleration, install a CUDA-enabled PyTorch build **after** installing this package.

Example for CUDA 12.6:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Usage

### Basic command

```bash
drone-bird-detector \
  --input-folder path/to/images \
  --input-gsd 0.005 \
  --output-folder path/to/results
```

### Arguments

| Argument | Description |
| -------- | ----------- |
| `--input_folder` | Path to a folder containing drone images (`.jpg`, `.png`, `.tif`). |
| `--input-gsd` | Ground sample distance (GSD) of the input images in **meters per pixel**. All images are assumed to have the same GSD. |
| `--box-score-thresh` | Minimum confidence score for a detection to be kept (default: `0.75`). When `--include-classes` is used, this threshold is applied **after class filtering** (and after renormalisation if enabled). |
| `--box-nms-thresh` | Intersection-over-union (IoU) threshold for non-max suppression. Lower values remove overlapping boxes more aggressively (default: `0.2`). |
| `--include-classes` | Path to a text file containing class names to include (one per line). Class names must exactly match those in `weights/labels.txt`. |
| `--renormalise` | Renormalise class scores after applying `--include-classes`. When enabled, scores represent **conditional probabilities given the included classes**, rather than absolute model confidence.|
| `--tile-width` | Width of image tiles in pixels (default: `1200`). |
| `--tile-height` | Height of image tiles in pixels (default: `1200`). |
| `--overlap` | Overlap between adjacent tiles in pixels (default: `400`). |
| `--device` | Device used to run inference (`cpu` or `cuda`, default: `cpu`). |
| `--batch_size` | Number of image tiles processed simultaneously by the model (default: `4`). |

### Example output CSV

```csv
image,species,confidence,xmin,ymin,xmax,ymax, background, class_1, class_2, ...
img_001.jpg,Red Knot,0.87,120,50,180,110,0.01,0.05,0.02,...
img_002.jpg,Bar-tailed Godwit,0.91,200,80,260,140,0.02,0.01,0.01,...
```

Each row represents a detected bird with:

* `image`: image filename
* `species`: predicted species
* `confidence`: model confidence (0–1)
* `xmin, ymin, xmax, ymax`: bounding box coordinates
* `background`: background model confidence
* `class_1`: class_1 model confidence
* `class_2`: class_2 model confidence

---

## Model Weights

The trained model weights are **not included in this repo**. They are available at:

* https://huggingface.co/JoshuaWilson/drone-bird-detector

The CLI will automatically download the weights the first time you run inference.

---

## Dataset

The annotated drone imagery dataset used to train this model is released under **CC BY 4.0** and available at:

* https://doi.org/10.48610/27809f1

Please cite both the paper and the dataset when using this tool.

---

## Citation

```bibtex
@article{Wilson2026,
  title={Big Bird: A global dataset of birds in drone imagery annotated to species level},
  author={Joshua P. Wilson and Tatsuya Amano and Thomas Bregnballe and Alejandro Corregidor-Castro and Roxane Francis and {Diego Gallego-García} and Jarrod C. Hodgson and Landon R. Jones and {César R. Luque-Fernández} and Dominik Marchowski and John McEvoy and Ann E. McKellar and W. Chris Oosthuizen and Christian Pfeifer and Martin Renner and {José Hernán Sarasola} and {Mateo Sokač} and Roberto Valle and Adam Zbyryt and Richard A. Fuller},
  journal={Remote Sensing in Ecology and Conservation},
  year={2026},
  doi={10.1002/rse2.70059}
}
```

---

## Licence

* **Code & CLI:** MIT License
* **Model weights:** MIT License
* **Dataset:** CC BY 4.0

---

## Recommended Workflow

1. Clone or install the package.
2. Place your drone images in a folder.
3. Run `drone-bird-detect` with the folder path.
4. Manually review a proportion of the results to validate the model's performance within the context of your dataset.
5. Use the resulting CSV for ecological analysis, transect summaries, or reporting.

---

## Contributing

Contributions are welcome. Please open an issue or pull request for:

* Bug fixes
* Improvements to CLI
* Adding new model architectures
* Enhancing output formats

Please respect the licence of the code, model, and dataset when contributing.

---

## Acknowledgements

This work was supported by The Moreton Bay Foundation, the Queensland Wader Study Group, and the Queensland Parks and Wildlife Service.

Thanks to Amanda M. Bishop, Cecilia Soldatini, Yuri V. Albores-Barajas, David W. Johnston, Madeline C. Hayes, Heather J. Lynch, Katarzyna Fudala, Robert Józef Bialik, Kyle H. Elliott, Marina Jiménez-Torres, Mauricio Soto-Gamboa, Marta Nowak, Olga Alexandrou, Samantha A. Collins, Orien M. W. Richmond, William T. Bean, and Sharon Dulava for their help collecting drone imagery.

Thanks to Sophie Rawson, Nicola Sockhill and Samantha Wong-Topp for their help in labelling the dataset.
