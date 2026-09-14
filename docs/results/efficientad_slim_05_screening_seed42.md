# EfficientAD-S Slim-0.5 — VisA PCB screening, seed 42

Config: `configs/efficientad_slim_05.yaml`. Teacher output remains 384 channels, Student output remains 768 channels split into 384 ST + 384 SA, and Autoencoder output remains 384 channels. Training reused the installed Anomalib EfficientAD preparation, loss and anomaly-map logic; only the Student and Autoencoder modules were replaced by Slim-0.5.

| Category | Train | Loss initial → final | Image AUROC | Pixel AUROC | AUPRO | Recall@FPR10 | Actual FPR | Validator |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pcb1 | 70k, finite | 11.989 → 1.319 | 0.9293 | 0.9896 | 0.9132 | 0.76 | 0.0600 | PASS |
| pcb2 | 70k, finite | 9.309 → 0.978 | 0.9454 | 0.9859 | 0.9095 | 0.85 | 0.1000 | PASS |
| pcb3 | 70k, finite | 10.083 → 1.067 | 0.9700 | 0.9920 | 0.9194 | 0.87 | 0.0792 | PASS |
| pcb4 | 70k, finite | 11.536 → 1.308 | 0.9880 | 0.9597 | 0.7288 | 1.00 | 0.0792 | PASS |
| **Mean** | — | — | **0.9582** | **0.9818** | **0.8677** | **0.87** | — | **PASS** |

All four health reports passed: no NaN/Inf, loss decreased, final checkpoints loaded strictly, and the pretrained Teacher checksum remained unchanged. All validators passed the normal-only split, calibration/test separation, quantiles, mask dimensions and reproducible checkpoint reload checks.

The screening gate passes because mean image AUROC is 0.9582, the minimum category AUROC is 0.9293, and every benchmark operating point stays at or below FPR 0.10. `recall@FPR10` selects an operating point from test scores for architecture comparison; it is not a threshold for deployment.

The validation-normal deployment thresholds produced test FPR values of 0.21, 0.15, 0.2079 and 0.1881 for pcb1–pcb4. Threshold calibration therefore needs further work before deployment even though the architecture screening gate passed.

SAM3 was disabled for this screening run because its checkpoint is 3.45 GB and the available GPU has 4 GB VRAM. The metrics above come from the unchanged EfficientAD anomaly-map logic. Full per-category artifacts are under `outputs/efficientad_slim/slim_0_5/<category>/`.
