# EfficientAD-S vs Slim-0.5 — VisA screening seed 42

`ΔAUROC = AUROC_slim - AUROC_old`, reported in percentage points (pp).

| Category | AUROC_old | AUROC_slim | ΔAUROC (pp) | Gate ≤2 pp |
|---|---:|---:|---:|:---:|
| pcb1 | 0.9506 | 0.9293 | -2.13 | FAIL |
| pcb2 | 0.9617 | 0.9454 | -1.63 | PASS |
| pcb3 | 0.9795 | 0.9700 | -0.95 | PASS |
| pcb4 | 0.9880 | 0.9880 | +0.00 | PASS |
| **Mean** | **0.9700** | **0.9582** | **-1.18** | — |

## Screening gate

- Student parameters: 56.52% reduction (target ≥40%): **PASS**
- Student fvcore-supported FLOPs: 62.33% reduction (target ≥30%): **PASS**
- AUROC degradation: ≤2 pp in every VisA category: **FAIL**
- Overall architecture screening: **FAIL**

The ≤1 pp objective belongs to the later real-PCBA/Jetson validation and is not evaluated or claimed here.

All AUROC values come directly from the existing `src/evaluator.py` output at `target_fpr: 0.10`; this comparison does not add or recompute a model metric.

## Implementation and literature roles

- [Batzner et al., EfficientAD](https://arxiv.org/abs/2303.14535): architecture and training source.
- [Anomalib EfficientAD](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/efficient_ad): FabLoop implementation source and pretrained Teacher loader.
- [Lee & Kim, arXiv:2407.17909](https://arxiv.org/abs/2407.17909): reference for EfficientAD feature distances and logical anomalies; it is not a channel-compression source.
- [rximg/EfficientAD](https://github.com/rximg/EfficientAD) and [DistillationAD](https://github.com/SimonThomine/DistillationAD): reading references only.
- [DepGraph/Torch-Pruning](https://github.com/VainF/Torch-Pruning): Plan B for structural pruning only if manual width reduction fails the quality gate.
- [CDA](https://github.com/WHUer-cloud/CDA): lightweight anomaly-distillation literature; it does not replace EfficientAD in this branch.
