# Behavioral Audit of Machine Unlearning Has a Privacy Cost

## 1. Requirements

- PyTorch >= 2.7.0 and torchvision
- CUDA >= 11.8.0 for GPU runs
- NumPy >= 2.2.5, scikit-learn, and tqdm
- pandas and matplotlib for plotting

## 2. Usage

### 2.1 Training

Use a fresh `--base_dir` for these experiments. Original surrogate training now includes the shared deletion set `D_u` and a sampled retained set. Earlier code trained originals without `D_u`, so old original or unlearned checkpoint caches must not be reused for the corrected deletion audit.

For CIFAR10 with ResNet18, 32 surrogates, and 500 randomly selected deletion records:

```bash
python main_train.py --dataset CIFAR10 --model ResNet18 --N 32 --un_size 500 --un_method random --base_dir ./save/cifar10-operational --seed 0
```

For Purchase100 with MLP4:

```bash
python main_train.py --dataset Purchase100 --model MLP4 --N 32 --un_size 500 --un_method random --base_dir ./save/purchase100-operational --seed 0
```

Purchase100 reserves 10,000 records using the dataset seed as a disjoint validation/query set before selecting training and deletion records. `dataset.train_source_idx` maps the reduced `full_set` indices to original Purchase100 record IDs; `dataset.valid_source_idx` maps validation indices. CIFAR uses its separate validation set. Checkpoint selection uses validation accuracy. NumPy and PyTorch are seeded with `seed + surrogate_id` before each surrogate is trained.

Training also accepts `--data_dir`, `--device`, `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, and `--overwrite`. Use `--un_method class` for deletion records drawn from class 0. The training environment is saved in `env_train.pkl` and original checkpoints in `orig/`.

### 2.2 Auditing

The operational audit fits separate compliance and retained-membership tests as described in Section 3.3, using the unmediated experimental setting in Section 4.1.

```bash
python single_audit.py --base_dir ./save/cifar10-operational --unlearn GradAscent --eta 0.5 --T_max 30 --repeats 30 --sigma 0.0 --seed 0
```

The reference uses the full unlearning schedule (`eta=1`). `--eta` selects the alternative's schedule fraction; it is not a calibrated measure of honesty. The example uses gradient ascent; `--unlearn SalUn` selects the existing SalUn implementation.

- `--target_idx` optionally selects a retained-membership candidate outside `D_u`. It indexes `dataset.full_set`; for Purchase100 this is the reduced training pool, not the original downloaded record numbering.
- `--T_max` defaults to 30. Each query budget from 1 through `T_max` has `--repeats` independently preselected query sequences (default 30).
- `--sigma` defaults to 0.0, measuring the training-ensemble baseline. A positive value adds independent Gaussian observation noise with that standard deviation to every posterior coordinate, using the same scale for both endpoint types. These measurements are not clipped or renormalized.
- `--seed` controls the audit's random choices. `--device` chooses the computation device; `--overwrite` recreates unlearned checkpoints within the selected training environment.

Queries are chosen from the validation/query pool independently of the candidate and model responses, before target access. Each response contains the full softmax vector. For each endpoint and trial, a transcript is generated once and supplied unchanged to both tests; the membership test makes no additional target queries. There is no mediator in this workflow, and the results are not a theorem-based privacy certificate.

Surrogate IDs are split 50%/25%/25% into fitting, calibration, and evaluation groups, stratified by the candidate's retained membership. Both update endpoints from an original surrogate stay in the same group. Each membership group needs at least four surrogate IDs to populate these splits; more are recommended because small held-out groups give coarse empirical rates. This is a data requirement, not an automatic validation step.

Separate diagonal Gaussian models are fitted to the compliance and membership populations. The membership model uses an equal mixture of reference and alternative endpoints in both its member and nonmember populations; it does not receive the unknown update label. Calibration chooses the threshold with minimum balanced error, breaking ties by proximity to zero. Fitted models and thresholds are frozen before held-out evaluation.

Compliance reporting uses `alpha` for accepting an alternative as the reference and `alpha_prime` for rejecting the reference. Membership reporting includes pooled and per-update results. Its measured signed advantage is `TPR - FPR`; the legacy `beta` column aliases this empirical attack quantity, not population total variation. Held-out rates come from actual endpoint transcripts, not samples generated from fitted Gaussian laws.

### 2.3 Saved results

Each audit creates a timestamped `results_.../` directory containing:

- `env_audit.pkl`: audit arguments, candidate labels, model-ID splits, and protocol metadata.
- `query_records.pkl`: the preselected query sequences for each budget and repetition.
- `trials/`: per-trial transcript arrays (`.npz`) and fitted tests/calibrated thresholds (`.pkl`).
- `heldout_predictions.csv`: held-out endpoint labels, scores, and decisions.
- `audit_results.csv`: measured compliance errors and membership metrics, including the legacy `beta` alias.
- `theory_results.csv`: Gaussian geometry diagnostics computed from the fitting split, separate from measured held-out performance.

Query repetitions describe variation across query choices and, when enabled, observation noise. They reuse the same underlying surrogate runs and are not independent model-run samples. Treat each trial as its own `T`-query audit; combining observations across trials would consume a larger query budget. Independently trained surrogate ensembles are needed to assess variation across independent model runs.

To plot saved audit results:

```bash
python plot_audit.py --base_dir ./save/cifar10-operational
```

The code changes have not been executed or tested, as requested.

## 3. Acknowledgements

Many of the code used in this repository are forked from the [code implementations](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry) of [Huang et al.](https://arxiv.org/abs/2409.19732), as noted in our paper. We thank the authors for making their code public.
