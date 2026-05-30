# Behavioral Audit of Machine Unlearning Has a Privacy Cost

## 1. Requirements

```
PyTorch >= 2.7.0
cuda    >= 11.8.0
numpy   >= 2.2.5
```

## 2. Usage

### 2.0 Convex Cases

To replicate Fig. 1, 2, 3, 5, 6, run the experiments from `standalone-convex` folder witht he default arguments.

### 2.1 Training

To train all `N` surrogate models, run: `python main_train.py --arguments`.

### 2.2 Auditing

To perform the (behavioral) audit, run: `python single_audit.py --base_dir {path to where the training environment will be stored} --unlearn {MU method} --eta {honesty level}`

You can specify `--seed` to ensure reproducibility

## 3. Acknowledgements

Many of the code used in this repository are forked from the [code implementations](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry) of [Huang et al.](https://arxiv.org/abs/2409.19732), as noted in our paper. We thank the authors for making their code public.