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

## Installation

### From PyPI (recommended)

```bash
pip install drone-bird-detector
```

### From GitHub (latest development version)

```bash
git clone https://github.com/joshwilson-dev/drone-bird-detector.git
cd drone-bird-detector
python -m venv venv         # optional but recommended
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
pip install -e .
```

---

## Usage

### Basic command

```bash
drone-bird-detector \
  --images path/to/images \
  --output results.csv \
  --confidence 0.5 \
  --device cpu
```

### Arguments

| Argument       | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `--images`     | Path to a folder containing drone images (`.jpg`, `.png`, etc.) |
| `--output`     | Path to the output CSV file (default: `results.csv`)            |
| `--confidence` | Minimum confidence threshold for detections (default: `0.5`)    |
| `--device`     | Device to run inference on (`cpu` or `cuda`)                    |

### Example output CSV

```csv
image,species,confidence,xmin,ymin,xmax,ymax
img_001.jpg,Red Knot,0.87,120,50,180,110
img_002.jpg,Bar-tailed Godwit,0.91,200,80,260,140
```

Each row represents a detected bird with:

* `image`: image filename
* `species`: predicted species
* `confidence`: model confidence (0–1)
* `xmin, ymin, xmax, ymax`: bounding box coordinates

---

## Model Weights

The trained model weights are **not included in this repo**. They are available at:

* https://huggingface.co/JoshuaWilson/drone-bird-detector

The CLI will automatically download the weights the first time you run inference.

---

## Dataset

The annotated drone imagery dataset used to train this model is released under **CC BY 4.0** and available at:

* [Dataset DOI or link]

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
4. Use the resulting CSV for ecological analysis, transect summaries, or reporting.

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