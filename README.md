# Large Scale Plastic Detection Pipeline via Machine Learning Augmented Remote Sensing

Code and data for my research paper of the same title: an end-to-end pipeline that takes raw
multispectral drone imagery and detects plastic litter in it, built to compare **how well
plastic can be detected in each spectral band**.

The pipeline trains one object detector per band on identical scenes (green, red, red edge,
near-infrared, and standard RGB) and then compares them statistically.

## Pipeline

```
raw drone captures ─► band alignment ─► labeling ─► chipping + augmentation ─► per-band training ─► evaluation + statistics
   5 sensors            ECC homography    boxes        694 chips per band        SSDLite detectors     PR/ROC, IoU, ANOVA
```

1. **Alignment** (`zcode/rawImageAlignment.py`): the drone's sensors are physically offset
   from each other, so the bands of a single capture do not line up. Each band is registered to a common
   frame with OpenCV's ECC algorithm using a homography model, parallelized across captures
   with a process pool.
2. **Chipping and augmentation** (`preprocess.py`): 134 labeled scenes are cut into
   detector-sized chips, with bounding boxes re-projected through every rotation, crop, flip,
   and brightness or contrast change. A custom `Img` class keeps each image and its boxes in
   sync through those transforms. Because the bands are aligned, the same chip and the same
   labels exist in all five bands.
3. **Train/test split** (`preprocess.py: prep`): split by image rather than by box, targeting
   70% of boxes in training, so no chip leaks across the split.
4. **Training** (`train.py`): one SSDLite320 MobileNetV3-Large detector per band, fine-tuned
   from pretrained weights at batch sizes 8, 16, and 32, with learning-rate reduction on
   plateau and early stopping.
5. **Evaluation** (`evaluate.py`): true and false positive counts, precision, recall, and F1
   swept across confidence thresholds, plus precision-recall and ROC curves, IoU distributions,
   prediction visualizations, and SHAP analysis.
6. **Statistics** (`zcode/anova.py`, `zcode/stats.py`): ANOVA and pairwise t-tests on detection
   confidence between band models, and per-band pixel-intensity analysis of plastic against
   background.

A second phase re-ran the comparison on Roboflow. The `roboflow/` scripts convert the chips
to JPEG and the label CSVs to Pascal VOC XML for upload.

## Dataset

| | |
| --- | --- |
| Source scenes | 134 multispectral drone captures |
| Bands | Green, Red, Red Edge, Near-Infrared, RGB |
| Chips | 694 per band (3,470 total) |
| Bounding boxes | 3,872 per band |
| Classes | bottle (2,723), can (786), container (363) |

The chips, labels, and trained weights are included in the repository, which is why it is
large (about 1.4 GB).

## Repository layout

```
terminal.py              entry point: runs preprocessing, training, and evaluation
preprocess.py            chipping, augmentation, dataset splits
train.py                 detector training
evaluate.py              metrics, curves, visualizations
evaluate/
  metrics.csv            results for every model at every confidence threshold
  train_*.csv, test_*.csv  the splits used
  b8_pth/, b16_pth/, b32_pth/   trained weights by batch size
new_data/chips/          chips_G, chips_NIR, chips_R, chips_RE, chips_RGB and label CSVs
roboflow/, ROBOFLOWEXPORT/     phase 2 conversion scripts and export
zcode/                   alignment, statistics, and supporting scripts
```

## Running it

```bash
pip install torch torchvision numpy pandas pillow scikit-learn scikit-image matplotlib seaborn shap scipy opencv-python rawpy piexif
python terminal.py
```

Set the project path in `DIRS` at the top of `terminal.py` to your clone location, then call
the stage you want (`do_preprocessing`, `do_training`, `do_evaluation`, or `quick_test` for a
five-epoch smoke test on one band).

## Author

Siddharth Nair. All code is my own; the multithreaded alignment script was written with AI
assistance, as noted in [`index.txt`](index.txt).
