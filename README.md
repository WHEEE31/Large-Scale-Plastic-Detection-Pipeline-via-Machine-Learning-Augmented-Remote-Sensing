# Large Scale Plastic Detection Pipeline via Machine Learning Augmented Remote Sensing

Code and data for my research paper of the same title ([read the full paper](paper.pdf)).
The project asks two questions:
**which spectral band makes plastic litter easiest to detect from a drone**, and can that
answer be turned into a detector accurate enough for real use?

## Key findings

- **Green beats near-infrared.** Across five spectral bands, the green band (560 nm) produced
  the most effective detector, significantly better than near-infrared (p < .05). This
  contradicted my hypothesis and the prior literature, which favoured NIR.
- **Why:** plastic stands out most against the background in green. Vegetation and dry leaf
  litter scatter near-infrared strongly, which hides the plastic. Green had the highest
  plastic-to-background pixel intensity ratio of any band.
- **The final detector clears the benchmark.** Trained on green-band imagery only, the Phase II
  model reached **88.8% precision, 83.0% recall, and an F1 score of 0.86**, above the 0.85
  threshold used in the literature for real-world object detection.

## Phase I: which band is best?

One detector was trained per band on identical scenes, so that the only difference between
models was the wavelength. The model acted as a proxy for how detectable plastic is in each band.

| Band | Wavelength | Effectiveness | Plastic-to-background intensity ratio |
| --- | --- | --- | --- |
| **Green** | 560 nm | **0.445** | **2.51** |
| Red | 650 nm | 0.392 | 1.88 |
| Visible (RGB) | — | 0.385 | 1.45 |
| Red edge | 730 nm | 0.345 | 1.63 |
| Near-infrared | 860 nm | 0.339 | 1.53 |

Effectiveness is a weighted average of accuracy (0.1), precision (0.25), recall (0.25), and
F1 (0.4), averaged over three training runs. Bands were compared with paired t-tests on
prediction confidence.

The Phase I models were weak in absolute terms. None reached the 0.85 benchmark, all had high
false-negative rates, and none correctly detected a single object in the Can class. They were
good enough to rank the bands, but not to use, which is what motivated Phase II.

## Phase II: building the detector

Using only the green-band data, I trained an RF-DETR (medium) detection transformer through
Roboflow on 1,754 images, with blur augmentation added to mimic motion during flight.

| Class | Average precision |
| --- | --- |
| Can (HDPE jugs) | 0.97 |
| Container (PET) | 0.86 |
| Bottle (PET) | 0.82 |

Training took 1 hour 28 minutes, against more than 17 hours for Phase I. Inference takes
183 ms per image; at a 50 m flight height the drone would have to exceed 583 m/s before the
model became the bottleneck, so the detector can keep up with any real flight.

## Pipeline

```
drone capture ─► band alignment ─► labeling ─► UniChip ─► per-band training ─► evaluation ─► green-band detector
 DJI Mavic 3M     ECC homography    labelImg    chipping +   SSD-MobileNet v3     t-tests,       RF-DETR medium
 at 50 m                                        jittering    (Phase I)            intensity      (Phase II)
```

1. **Capture:** 223 plastic objects (PET bottles, HDPE jugs, PET containers) were laid out on
   grass and asphalt at the Lamont-Doherty Earth Observatory and photographed from 50 m with a
   DJI Mavic 3M, giving four multispectral bands plus visible light for each of 134 scenes.
2. **Alignment** (`zcode/rawImageAlignment.py`): the drone's cameras are physically offset, so
   the bands of one capture do not line up. Each band is registered to a common frame with
   OpenCV's ECC algorithm using a homography model, parallelized with a process pool.
3. **UniChip** (`preprocess.py`): my preprocessing algorithm. It crops scenes into chips with a
   sliding window that avoids cutting objects in half, then augments them with rotations,
   shifts, and brightness changes. Every transform is applied identically across all five
   bands, so corresponding chips contain the same objects in the same places. That guarantee
   is what makes the band comparison fair.
4. **Split** (`preprocess.py: prep`): 70% training and 30% testing, split by image rather than
   by box so no chip appears on both sides.
5. **Phase I training** (`train.py`): one SSDLite MobileNetV3 detector per band, fine-tuned
   from COCO weights, with learning-rate reduction on plateau and early stopping.
6. **Evaluation** (`evaluate.py`, `zcode/`): precision, recall, F1, and effectiveness swept
   across confidence thresholds from 0.05 to 0.95, confusion matrices, paired t-tests, and
   per-band pixel-intensity analysis.
7. **Phase II** (`roboflow/`): scripts that convert the green chips to JPEG and the labels to
   Pascal VOC XML for Roboflow, where the RF-DETR model was trained and evaluated.

## Dataset

| | |
| --- | --- |
| Source scenes | 134 multispectral drone captures |
| Bands | Green, Red, Red Edge, Near-Infrared, Visible |
| Phase I chips | 694 per band (3,470 total), 3,872 labeled objects per band |
| Classes | bottle (2,723), can (786), container (363) |
| Phase II images | 1,754 green-band images after Roboflow augmentation |

The chips, labels, and Phase I weights are included in the repository, which is why it is
large (about 1.4 GB).

## Repository layout

```
paper.pdf                the full research paper
terminal.py              entry point: runs preprocessing, training, and evaluation
preprocess.py            UniChip: chipping, augmentation, dataset splits
train.py                 Phase I detector training
evaluate.py              metrics, curves, visualizations
evaluate/
  metrics.csv            results for every model at every confidence threshold
  train_*.csv, test_*.csv  the splits used
  b8_pth/, b16_pth/, b32_pth/   trained weights by batch size
new_data/chips/          chips_G, chips_NIR, chips_R, chips_RE, chips_RGB and label CSVs
roboflow/, ROBOFLOWEXPORT/     Phase II conversion scripts and export
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

## Limitations and next steps

- Images were taken while hovering. Continuous flight will add motion blur beyond what the
  augmentation covers.
- Only two polymers (PET and HDPE) and two backgrounds (grass and asphalt) were tested, in
  late autumn.
- Transparent PET bottles remain the hardest class. Shorter wavelengths such as ultraviolet
  may separate them better.
- The next step is an end-to-end system that turns detections into a geolocated litter map
  using the drone's image metadata.

## Author

Siddharth Nair. I designed and carried out the study and wrote all of the code; the
multithreaded alignment script was written with AI assistance, as noted in
[`index.txt`](index.txt). A mentor advised on scope and provided the drone and flight permissions.
