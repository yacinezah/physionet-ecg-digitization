# PhysioNet ECG Image Digitization

Portfolio project for the Kaggle / PhysioNet ECG Image Digitization challenge: reconstruct 12-lead ECG time-series signals from scanned or photographed ECG paper images.

This repository documents a complete computer-vision and signal-processing workflow: image normalization, ECG grid rectification, trace extraction, pseudo-mask generation, cross-lead attention fine-tuning, and final inference.

## Problem

Input: an ECG image.  
Output: numerical 12-lead ECG waveforms.  
Metric: signal-to-noise ratio (SNR) between reconstructed and reference waveforms after competition alignment.

The task is not only segmentation. Small geometric errors in grid rectification produce amplitude and timing errors in the recovered signal, so the pipeline combines computer vision, geometric calibration, segmentation, signal reconstruction, and post-processing.

## Pipeline

1. Stage 0: coarse image normalization and homography.
2. Stage 1: ECG grid detection and rectification.
3. Stage 2: trace extraction from rectified lead-row crops.
4. Pixel-to-signal decoding and submission formatting.

## Repository structure

```text
notebooks/
  01_generate_pseudo_masks.ipynb
  02_train_cross_attention.ipynb
  03_inference_submission.ipynb

src/
  stage2_lead_model.py

reports/
  main.tex
  report.pdf

models/
  README.md

docs/
  project_summary.md
```

## My contributions

- Reproduced and studied a strong three-stage public baseline for ECG image digitization.
- Built a pseudo-label generation pipeline that saves rectified images and sparse COO trace masks.
- Implemented a modified Stage 2 lead model with no-fusion, Conv2D, shared Conv2D, Conv3D, and cross-lead attention modes.
- Trained the cross-attention model under Kaggle GPU constraints using progressive unfreezing and checkpoint resume logic.
- Tested first-place-inspired Stage 1 ideas: sub-pixel grid localization and polynomial surface fitting.
- Compared variants honestly and kept the simpler configuration when the complex changes did not improve final SNR.

## Results

| Configuration | Private LB | Public LB |
|---|---:|---:|
| Best final variant: no attention, no Stage 1 surface-fitting improvement | **23.27649** | **23.37199** |
| One whole model + one lead model, no attention, no Stage 1 improvement, no TTA | 23.02462 | 23.24447 |
| Stage 1 surface-fitting improvement, one model, no TTA | 22.03615 | 22.20997 |
| Best visible attention variant, no Stage 1 improvement | 18.52220 | 18.72364 |

Main lesson: the cross-attention branch trained successfully, but it did not beat the simpler baseline in end-to-end leaderboard score. This is a useful negative result and is discussed in the report.

## How to run

These notebooks are designed for Kaggle, not for direct local execution. They require the official competition dataset, public baseline checkpoints, the custom `stage2_lead_model.py`, and GPU acceleration.

Typical order:

1. Run `notebooks/01_generate_pseudo_masks.ipynb`.
2. Run `notebooks/02_train_cross_attention.ipynb`.
3. Run `notebooks/03_inference_submission.ipynb`.

## Credits

This project builds on public Kaggle work and solution writeups, especially the hengck23 demonstration pipeline and high-ranking Stage 2 lead-model ideas. My contribution is the reproduction, modification, pseudo-label workflow, attention experiment, ablation analysis, and documentation.
