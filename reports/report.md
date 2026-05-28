# Project report summary

The full report is titled **Digitalisation d’ECG à partir d’images — Étude, modification et évaluation d’un pipeline Kaggle en trois stages**.

Authors: Zahouani Yacine and Allal Nabil.  
Supervisor: Marc Donias.  
Date: 11 May 2026.

## Scope

The report studies Task 1 of the PhysioNet / Kaggle ECG Image Digitization challenge: reconstructing 12-lead ECG waveforms from paper ECG images.

## Main pipeline

The project is organized around a three-stage pipeline:

1. **Stage 0:** geometric normalization using landmark detection, orientation prediction, and homography.
2. **Stage 1:** ECG grid detection, grid-point reconstruction, interpolation, and local homography rectification.
3. **Stage 2:** waveform extraction from rectified images using whole-image models and lead-row models.

## Main contributions

- Stage 1 sub-pixel grid localization with weighted connected-component centroids.
- Stage 1 polynomial surface fitting for global grid regularization.
- Sparse COO pseudo-mask generation for memory-efficient training.
- Modified Stage 2 LeadModel with cross-lead attention fusion.
- Progressive unfreezing and checkpointed Kaggle training for the attention model.
- Ablation study showing that the final best variant is the simpler no-attention/no-surface-fitting configuration.

## Main result

| Configuration | Private LB | Public LB |
|---|---:|---:|
| Best final variant: no attention, no Stage 1 surface fitting | 23.27649 | 23.37199 |
| Simple 1 whole + 1 lead, no attention, no Stage 1 improvement, no TTA | 23.02462 | 23.24447 |
| Stage 1 surface-fitting variant | 22.03615 | 22.20997 |
| Best visible attention variant | 18.52220 | 18.72364 |

## Conclusion

The project shows a full experimental engineering workflow: understand a strong baseline, implement targeted improvements, train under resource constraints, and keep the configuration that is actually supported by the metric.

The most important lesson is that technically plausible improvements, such as cross-lead attention, do not automatically improve the final competition score when pseudo-label quality, compute budget, and end-to-end integration become the limiting factors.
