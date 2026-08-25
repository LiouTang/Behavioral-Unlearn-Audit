# Behavioral Audit of Machine Unlearning Has a Privacy Cost

## 1. Requirements

```
PyTorch >= 2.7.0
cuda    >= 11.8.0
numpy   >= 2.2.5
```

## 2. Usage

### 2.1 Training

To train all `N` surrogate models, run:
```
python main_train.py --dataset {dataset} --model {model} --N {N} --un_size {size of D_u} --un_method {random / class}
                     --batch_size {batch size} --num_epochs {training epochs} --lr {learning rate} --weight_decay {weight decay}
                     --base_dir {path to where the training environment will be stored} --seed {seed for reproducibility}
                     --overwrite {overwrite trained models to start fresh}
```

### 2.2 Auditing

To perform the (behavioral) audit, run:
```
python single_audit.py --base_dir {path to where the training environment will be stored}
                       --target_idx {OPTIONAL, id of target z^*} --unlearn {MU method}
                       --eta {honesty level} --T_max {maximum query budget}
                       --seed {seed for reproducibility} --overwrite {overwrite unlearned models to start fresh}
```

You can specify `--seed` to ensure reproducibility

### 2.3 Convex Models

To replicate Figs on convex models, run the experiments from `standalone-convex` folder with the default arguments.

## 3. Acknowledgements

Many of the code used in this repository are forked from the [code implementations](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry) of [Huang et al.](https://arxiv.org/abs/2409.19732), as noted in our paper. We thank the authors for making their code public.