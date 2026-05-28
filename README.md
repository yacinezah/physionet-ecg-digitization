# PhysioNet ECG Image Digitization

Portfolio project for the Kaggle / PhysioNet ECG Image Digitization challenge: reconstruct 12-lead ECG time-series signals from scanned or photographed ECG paper images.

This repository documents a complete computer-vision and signal-processing workflow: geometric normalization, ECG grid rectification, trace extraction, sparse pseudo-mask generation, cross-lead attention fine-tuning, and final inference.

## Problem

**Input:** an ECG image.  
**Output:** numerical 12-lead ECG waveforms.  
**Metric:** signal-to-noise ratio (SNR) after competition alignment.

The task is not only segmentation. Small geometric errors in grid rectification produce amplitude and timing errors in the recovered signal, so the pipeline combines computer vision, geometric calibration, segmentation, signal reconstruction, and post-processing.

## Pipeline

```text
Raw ECG image
  -> Stage 0: document normalization and homography
  -> Stage 1: ECG grid detection and rectification
  -> Stage 2: trace segmentation from rectified lead crops
  -> Pixel-to-signal decoding
  -> Kaggle submission.csv
```

## Repository structure

```text
notebooks/
  01_generate_pseudo_masks.ipynb     # pseudo-label generation
  02_train_cross_attention.ipynb     # attention fine-tuning
  03_inference_submission.ipynb      # inference notebook placeholder / export target

src/
  stage2_lead_model.py               # modified Stage 2 LeadModel

reports/
  report.md                          # report summary
  report.pdf                         # final PDF report, add manually if needed

models/
  README.md                          # external checkpoint notes

docs/
  project_summary.md
```

## My contributions

- Reproduced and studied a strong three-stage public ECG digitization baseline.
- Built a pseudo-label generation pipeline that saves rectified images and sparse COO trace masks.
- Implemented a modified Stage 2 LeadModel with cross-lead feature fusion and attention.
- Trained the attention model under Kaggle GPU constraints using progressive unfreezing and checkpoint resume logic.
- Tested Stage 1 sub-pixel grid localization and polynomial surface fitting.
- Ran ablations and kept the simpler configuration when the complex variants did not improve final SNR.

## Results

| Configuration | Private LB | Public LB |
|---|---:|---:|
| Best final variant: no attention, no Stage 1 surface-fitting improvement | **23.27649** | **23.37199** |
| One whole model + one lead model, no attention, no Stage 1 improvement, no TTA | 23.02462 | 23.24447 |
| Stage 1 surface-fitting improvement, one model, no TTA | 22.03615 | 22.20997 |
| Best visible attention variant, no Stage 1 improvement | 18.52220 | 18.72364 |

Main lesson: the cross-attention branch trained successfully, but it did not beat the simpler baseline in end-to-end leaderboard score. This is a useful negative result: a more expressive model is not automatically better when pseudo-label quality, memory, training time, and integration stability are limiting factors.

## How to run

These notebooks are designed for Kaggle, not for direct local execution. They require:

1. the official competition dataset,
2. public baseline checkpoints,
3. the custom `src/stage2_lead_model.py`, copied or packaged as a Kaggle dataset,
4. GPU acceleration.

Typical order:

1. Run `notebooks/01_generate_pseudo_masks.ipynb`.
2. Run `notebooks/02_train_cross_attention.ipynb`.
3. Replace `notebooks/03_inference_submission.ipynb` with the full Kaggle export if reproducing the final submission.

## Credits

This project builds on public Kaggle work and solution writeups, especially the hengck23 demonstration pipeline and high-ranking Stage 2 lead-model approaches such as Takashi Someya's solution. My contribution is the reproduction, modification, pseudo-label workflow, attention experiment, ablation analysis, and documentation.
