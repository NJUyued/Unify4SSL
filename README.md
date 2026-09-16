# Unify4SSL

A unified framework for semi-supervised learning (SSL) methods, refactoring five SSL codebases into one pluggable architecture. All methods share a FixMatch-style backbone (weak/strong augmentation, pseudo-labels, EMA) and are implemented as composable strategies over shared algorithmic components.

## Methods

| Method | Paper | Venue | Core idea | Setting |
|---|---|---|---|---|
| FixMatch | Sohn et al., *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence* | NeurIPS 2020 | confidence-thresholded pseudo-labels (baseline anchor) | classic SSL |
| RDA | Duan et al., *Reciprocal Distribution Alignment for Robust Semi-supervised Learning* | ECCV 2022 | dual-head reciprocal distribution alignment, no threshold | distribution mismatch |
| PRG | Duan et al., *Towards Semi-supervised Learning with Non-random Missing Labels* | ICCV 2023 | class-transition-tracking pseudo-label rectification (Markov random walk) | MNAR missing labels |
| SoC | Duan et al., *Roll with the Punches: Expansion and Shrinkage of Soft Label Selection for Semi-Supervised Fine-Grained Visual Classification* | AAAI 2024 | confidence-aware multi-granularity soft-label selection | fine-grained SSL |
| MutexMatch | Duan et al., *MutexMatch: Semi-Supervised Learning with Mutex-Based Consistency Regularization* | TNNLS 2024 | high/low-confidence mutex consistency with a complementary-label head | classic SSL |
| USP | Duan et al., *Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-Supervised Continual Learning* | ICCV 2025 | FSR (ETF anchors) + DCP (classifier/prototype routing) + CUD (unlabeled distillation) | semi-supervised continual learning |

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# FixMatch, CIFAR-10 with 40 labels
python tools/train.py -c configs/fixmatch/cifar10.yaml

# PRG under the MNAR (PRG protocol) setting
python tools/train.py -c configs/prg/cifar10_mnar_prg.yaml

# SoC on Semi-Aves (download dataset first, see below)
python tools/train.py -c configs/soc/semi_aves.yaml

# USP on CIFAR-100 class-incremental
python tools/train.py -c configs/usp/cifar100.yaml
```

CLI overrides beat yaml values:

```bash
python tools/train.py -c configs/prg/cifar10.yaml --num_labels 250
```

## Architecture

```
sslunify/
├── core/           # unified training skeleton (~80% of the original boilerplate)
│   ├── trainer.py      # SSLTrainer: iteration loop, AMP, EMA, eval cadence, checkpointing
│   ├── losses.py       # ce_loss (hard/soft), FixMatch consistency
│   ├── optim.py        # SGD with BN weight-decay skip
│   ├── schedulers.py   # cosine schedule with warmup
│   └── ema.py, meters.py
├── data/           # datasets, lb/ulb splits, MNAR / label-noise protocols, samplers
├── nets/           # WRN-28-2, CIFAR ResNet-18, torchvision ResNet-50/101, CNN-13, PreResNet
├── components/     # ★ shared algorithmic components (the core of the unification)
│   ├── ctt.py          # ClassTransitionTracker: C×C×N transition matrix + label bank
│   │                   #   (shared verbatim by PRG and SoC in the original repos)
│   ├── dist_queue.py   # sliding-window batch-mean distribution queue (RDA, PRG, USP-DistAlign)
│   ├── heads.py        # ReverseCLS complementary head + complementary-label utilities
│   │                   #   (shared by MutexMatch and RDA)
│   └── kmeans.py       # multi-granularity k-means over the class-affinity matrix (SoC)
├── methods/        # the five methods as pluggable strategies
│   ├── base.py         # SSLMethod interface: build_models / train_step / extra_metrics
│   ├── fixmatch.py     # baseline anchor
│   ├── prg.py, rda.py, mutexmatch.py, soc.py
│   └── usp.py          # loss definitions for continual learning (driven by cl.SessionTrainer)
└── cl/             # continual-learning scaffolding for USP
    └── session.py      # SessionTrainer: session loop, expandable head, prototypes, exemplar memory
```

### Design

Each original repo shipped its own copy of the same FlexMatch-style training loop
(data iteration, EMA, cosine schedule, evaluation, checkpointing — nearly byte-identical
across repos). In Unify4SSL:

- **`SSLTrainer` owns the loop.** A method implements `train_step` (one iteration:
  forward + state update + losses) and hooks for model structure and extra metrics.
- **Shared state becomes components.** PRG and SoC both track class transitions in a
  C×C×N sliding window (identical code in both repos); here it is one
  `ClassTransitionTracker`, consumed two ways (Markov-walk pseudo-label rectification
  vs. multi-granularity clustering). RDA and MutexMatch both use a `ReverseCLS`
  complementary head with different complementary-label sources (random vs. argmin).
- **USP keeps its own outer loop.** Continual learning is structurally a two-level loop
  (sessions × epochs) with cross-session state (reference model, prototypes, exemplar
  buffer, expandable classifier), so it lives in `cl/SessionTrainer` and reuses the
  data pipeline and loss utilities instead of being forced into the single-task loop.

### Faithfulness

Losses are ported verbatim from the original implementations, including behaviors that
look like bugs but produce the published numbers, e.g.:

- RDA's sliding-window distributions are initialized to **ones**, not zeros;
- RDA's `normalize_d` normalizes over the *entire batch tensor* (rows do not sum to 1);
- RDA's temperature parameter is unused in its consistency loss (template vestige);
- MutexMatch's reverse loss is computed on detached features over the *full* batch,
  while RDA's is computed on non-detached features over *labeled* data only.

Such points are marked in docstrings. Deviations are listed in each method's docstring.

## Datasets

- CIFAR-10/100, SVHN, STL-10: auto-downloaded via torchvision to `--data_dir`.
- mini-ImageNet / Tiny-ImageNet: place under `data_dir` following the original repo layout.
- Semi-Aves / Semi-Fungi (SoC): download from the
  [SS-FGVC benchmark](https://github.com/cvl-umass/semi-fgvc) and point `--data_dir` at it.
- CIFAR-100 / ImageNet-100 class-incremental splits (USP) are generated on the fly.

## Citation

If you use the underlying methods, please cite the original papers:

```bibtex
@inproceedings{duan2022rda,
  title={RDA: Reciprocal Distribution Alignment for Robust Semi-supervised Learning},
  author={Duan, Yue and Shi, Yinghuan and Liu, Zhen and Wang, Jiayi and Gao, Yuan and Tong, Linwei and Lai, Zhihui},
  booktitle={ECCV},
  year={2022}
}
@inproceedings{duan2023prg,
  title={Towards Semi-supervised Learning with Non-random Missing Labels},
  author={Duan, Yue and Shi, Yinghuan and Liu, Zhen and Wang, Jiayi and Zhang, Xizhou and Tong, Linwei and Gao, Yuan and Yang, Jian},
  booktitle={ICCV},
  year={2023}
}
@inproceedings{duan2024soc,
  title={Roll with the Punches: Expansion and Shrinkage of Soft Label Selection for Semi-Supervised Fine-Grained Visual Classification},
  author={Duan, Yue and Shi, Yinghuan and Wang, Jiayi and Liu, Zhen and Zhang, Li and Yang, Jian},
  booktitle={AAAI},
  year={2024}
}
@article{duan2024mutexmatch,
  title={MutexMatch: Semi-Supervised Learning with Mutex-Based Consistency Regularization},
  author={Duan, Yue and Shi, Yinghuan and Wang, Jiayi and Liu, Zhen and Yin, Xizhou and Yang, Jian},
  journal={IEEE TNNLS},
  year={2024}
}
@inproceedings{duan2025usp,
  title={Divide-and-Conquer for Enhancing Unlabeled Learning, Stability, and Plasticity in Semi-Supervised Continual Learning},
  author={Duan, Yue and others},
  booktitle={ICCV},
  year={2025}
}
```
