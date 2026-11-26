# Behavioral Audit of Machine Unlearning Has a Privacy Cost

## 1. Requirements

- PyTorch >= 2.7.0 and torchvision
- CUDA >= 11.8.0 for GPU runs
- NumPy >= 2.2.5, scikit-learn, tqdm, and pandas
- matplotlib for plotting

## 2. Training environments and checkpoint reuse

The declared deletion experiment requires each original model to have been trained on the shared deletion set `D_u` together with its sampled retained set `D_r`. The corrected trainer includes both sets.

Existing checkpoints do not need regeneration merely because the audit now includes mediation. Original models that actually trained on `D_u` plus `D_r`, and their matching reference/alternative unlearning endpoints, can be reused with their original training environment and dataset-index metadata. The supplied old training code instead excluded `D_u` from every original model. Checkpoints produced by that exact path cannot serve as pre-deletion originals for the declared `D_u`; they may serve as retained-only training references if their training and checkpoint-selection procedure matches the declared reference. Endpoints obtained by updating those faulty originals are not endpoints for deleting previously trained records in `D_u`.

Keep valid old weights with their original `env_train.pkl`. Do not run the trainer into an existing checkpoint directory to regenerate metadata: a newly generated environment can give old indices different meanings. Use a fresh `--base_dir` when new original models are needed. This code does not automatically establish checkpoint provenance.

For CIFAR10 with ResNet18, 32 surrogates, and 500 randomly selected deletion records:

```bash
python main_train.py --dataset CIFAR10 --model ResNet18 --N 32 --un_size 500 --un_method random --base_dir ./save/cifar10-operational --seed 0
```

For the local Purchase100/MLP4 run, use 32 models and the paper's 1,000-record deletion set:

```bash
python main_train.py --dataset Purchase100 --model MLP4 --N 32 --un_size 1000 --un_method random --base_dir ./save/purchase100-local --seed 0
```

New Purchase100 environments reserve 10,000 seeded records as a disjoint validation/query set before choosing training and deletion records. `dataset.train_source_idx` maps the reduced `full_set` indices to original Purchase100 record IDs; `dataset.valid_source_idx` maps validation indices. The old Purchase100 environment used the training population as its validation pool. Otherwise valid old weights could be audited using genuinely unused external query data, but reserving records after training does not make them retrospectively held out. The automatic reservation applies only to newly constructed environments.

CIFAR uses its separate validation set. The current trainer selects checkpoints by validation accuracy and seeds NumPy and PyTorch with `seed + surrogate_id`. Older models selected by training accuracy are not automatically exact retraining references for this changed selection rule.

Training also accepts `--data_dir`, `--device`, `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, and `--overwrite`. Use `--un_method class` for deletion records drawn from class 0. The training environment is saved in `env_train.pkl` and original checkpoints in `orig/`.

## 3. Mediated operational audit

The evaluation implements Section 3.3's fixed-query protocol: a mediator publishes its channel policy before target access, receives the observation transcript, and releases one sanitized transcript. Separate compliance and retained-membership tests reuse that same release. The evaluator compares an unmediated baseline with Gaussian and independent-coordinate Laplace mediator noise.

### 3.1 Sweep mediator noise levels

```bash
python single_audit.py --base_dir ./save/purchase100-local --unlearn GradAscent --eta 0.5 --T_max 30 --repeats 30 --sigma 0.05 --mediators gaussian laplace --noise_scales 0.1 0.5 1.0 2.0 5.0 --seed 0
```

`--mediators` selects either or both of `gaussian` and `laplace`; the default includes both. `--noise_scales` accepts one or more levels and defaults to `0.1 0.5 1.0`. Each level defines a separate protocol, and an unmediated baseline is always included.

- For Gaussian mediation, `noise_scale` is the per-coordinate standard deviation `sigma_M`.
- For Laplace mediation, `noise_scale` is the per-coordinate Laplace scale `ell`, whose variance is `2 * ell**2`. Equal numeric scales do not mean equal variances.
- `--sigma` is the independent Gaussian **observation** standard deviation, before mediation. It is distinct from the mediator's scale. The default `0.0` gives a raw-probability input variant; the examples use a positive value to retain the paper's explicit observation channel.

Both noises act independently across the complete `k*T` response coordinates. Neither observations nor releases are clipped or renormalized. All endpoint types use the same declared channel.

### 3.2 Compare at fixed certified privacy budgets

Supply `--privacy_budgets` to add policies whose mediator scale is calibrated separately for each query budget `T`:

```bash
python single_audit.py --base_dir ./save/purchase100-local --unlearn GradAscent --eta 0.5 --T_max 30 --repeats 30 --sigma 0.05 --mediators gaussian laplace --noise_scales 0.5 1.0 --privacy_budgets 0.05 0.1 0.2 0.5 --bootstrap_samples 2000 --seed 0
```

The scale-sweep policies and budget-calibrated policies are recorded separately. Budget calibration uses only the public response domain, `T`, observation scale, and requested budget. It does not use endpoint responses, target identity or membership, surrogate geometry, or observed audit accuracy. The supplied budgets must lie strictly between zero and one; scales are nonnegative.

### 3.3 Domain-wide certificates and the utility ceiling

For `T` probability-vector responses, the response domain is a product of simplices, with Euclidean diameter `sqrt(2*T)` and L1 diameter `2*T`.

For Gaussian mediation, the combined scale is

```text
s = sqrt(sigma**2 + sigma_M**2)
b_cert = 2 * Phi(sqrt(2*T) / (2*s)) - 1
```

This is the exact domain-wide Gaussian channel diameter from Section 3.3. For a requested budget `b`, the policy uses

```text
s_required = sqrt(2*T) / (2 * Phi_inverse((1+b)/2))
sigma_M = sqrt(max(0, s_required**2 - sigma**2))
```

For Laplace mediation, the evaluator uses a conservative domain-wide upper bound, not the Gaussian formula with a substituted variance. Two one-dimensional Laplace distributions of scale `ell`, shifted by `delta`, have overlap `exp(-abs(delta)/(2*ell))`. The product of coordinatewise overlaps lower-bounds the overlap of the product distributions, giving

```text
TV(Laplace(r, ell), Laplace(r_prime, ell))
    <= 1 - exp(-L1_distance(r, r_prime) / (2*ell))
    <= 1 - exp(-T/ell) = b_L.
```

Adding independent observation noise cannot increase TV. Conversely, adding Laplace noise to the observation channel cannot increase its Gaussian domain-wide bound. Therefore the combined channel is certified by

```text
b_obs = 2 * Phi(sqrt(2*T) / (2*sigma)) - 1
b_cert = min(b_obs, b_L).
```

Zero noise uses the corresponding limiting bound of one. A Laplace policy targeting `b` uses `ell = -T / log(1-b)`, or zero mediator scale if the observation channel already meets `b`. This is a sufficient scale from the stated bound; it is not claimed to be the smallest possible scale for the convolution. The Laplace certificate is an extension using the paper's general domain-wide proposition, not an assertion that its exact Gaussian formulas apply to Laplace responses.

For either mechanism, the reported certificate gives

```text
alpha + alpha_prime >= 1 - b_cert
balanced_audit_accuracy <= (1 + b_cert) / 2.
```

The Gaussian certificate is exact; the Laplace certificate may be loose. Both imply the stated ceiling. The certificate applies to the full released batch for every response pair, independently of the particular retained-membership attack fitted in the experiment. It is not a certificate obtained from low observed attack success.

### 3.4 Queries, fitted tests, and held-out evaluation

The reference uses the full unlearning schedule (`eta=1`). `--eta` selects the alternative's schedule fraction; it is not a calibrated measure of honesty. `--unlearn SalUn` selects the existing SalUn implementation instead of gradient ascent.

- `--target_idx` optionally selects a candidate outside `D_u`. It indexes `dataset.full_set`; for new Purchase100 environments, this is the reduced training pool, not the original downloaded record numbering.
- `--T_max` defaults to 30. Every query budget from 1 through `T_max` has `--repeats` preselected query sequences (default 30).
- `--seed` controls reproducibility, and `--device` chooses the computation device. `--overwrite` recreates unlearned checkpoints within the selected training environment; mediation itself does not retrain endpoint models.
- `--bootstrap_samples` defaults to 2000 and controls the saved empirical uncertainty summaries.

Queries are fixed from the disjoint validation/query pool independently of the candidate and model responses. Responses contain all class probabilities. Policies share model runs and query sequences for matched comparisons. Within each trial, each endpoint's released transcript is generated once and passed to both tests without additional target queries.

Surrogate IDs are split 50%/25%/25% into fitting, calibration, and evaluation groups, stratified by retained membership. Both update endpoints from one original model stay in the same group. Each membership group needs at least four surrogate IDs to populate these splits; more are recommended because small held-out groups give coarse rates. This is a data requirement, not an automatic validation step.

For each policy, separate diagonal Gaussian score models are fitted to its sanitized fitting transcripts. They remain fitted attack models even when the channel is Laplace; they do not assert that the true response laws are Gaussian or that the attack is optimal. The membership test uses an equal mixture of reference and alternative endpoints in both membership populations and never receives the target's unknown update label. Independent calibration chooses thresholds minimizing balanced error, with ties broken by proximity to zero. Both score models and thresholds are frozen before held-out scoring.

`alpha` is the rate of accepting an alternative as the reference, and `alpha_prime` is the rate of rejecting the reference. Membership reporting includes pooled and per-update results. Its signed advantage is `TPR - FPR`; the legacy `beta` column aliases this measured attack quantity, not population TV. Errors and advantages come from held-out model transcripts, not synthetic samples drawn from fitted Gaussian laws.

## 4. Deliverables and interpretation

Each audit creates `mediated_results_TIMESTAMP/` containing:

- `env_audit.pkl`: arguments, candidate labels, model-ID splits, and experiment metadata.
- `query_records.pkl`: preselected queries for each budget and repetition.
- `policies.csv`: public channel settings, requested budgets, achieved certificates, and implied utility ceilings/error floors for each `T` and policy.
- `audit_results.csv`: measured errors and membership metrics for each `T`, query repetition, and policy.
- `heldout_predictions.csv`: held-out endpoint identities, labels, scores, and decisions.
- `theory_results.csv`: fitting-split Gaussian geometry diagnostics for the unmediated baseline only. They are not transferred to mediated outputs as certified leakage lower bounds.
- `mediation_summary.csv`: per-policy summaries, conditional 95% bootstrap intervals, and paired utility loss relative to the unmediated baseline.

Full response transcripts and per-trial fitted Gaussian parameters are kept only in memory, not saved. The recorded scores, decisions, and thresholds support the reported metrics, bootstrap summaries, and figures without those large intermediate arrays. Retain the original dataset, training/index environment, endpoint weights, code revision, audit seed, query records, and policies to regenerate transcripts and refit tests when needed. Reproduction seeds remain trusted evaluator metadata.

An MLP4 checkpoint is approximately 20 MB. One 32-model GA comparison stores 32 original, 32 reference, and 32 alternative checkpoints: approximately 1.9 GB total. The shared Purchase100 training environment adds approximately 0.5 GB; CSV results, query metadata, and figures add much less. Additional mediator settings add tabular results, not another collection of transcript arrays or endpoint checkpoints. The local example evaluates only GA at fraction 0.5; the larger method/fraction grid is not required for an end-to-end run.

To generate the figures for completed audits in a training run folder:

```bash
python plot_audit.py --base_dir ./save/purchase100-mlp4-n32-20261124
```

`plot_audit.py` discovers both `results_*` and `mediated_results_*` folders. It also accepts a single result folder through either `--base_dir` or its `--result_dir` alias:

```bash
python plot_audit.py --result_dir ./save/purchase100-mlp4-n32-20261124/mediated_results_1124-144326-796100
```

For mediated results, it saves the original audit-error/membership-advantage tradeoff as `figures/audit_tradeoff.pdf` and `.png`, using only the unmediated baseline and its fitted diagnostics. It also generates the mediated PDF and PNG figures listed below under `figures/`. Use `--T` to select the query budget for mediated snapshot plots; the default is the largest saved budget. Legacy results retain their `Figs/{method}_{eta}.pdf` output under the training run folder. `python plot_mediation.py --result_dir <result_folder>` remains available when only the mediated figures are wanted.

1. Audit balanced accuracy and error sum versus certified privacy budget, with the ceiling `(1+b_cert)/2`, floor `1-b_cert`, and unmediated baseline.
2. Measured retained-membership advantage versus certified budget, pooled and separately for reference and alternative update procedures.
3. Audit utility versus query budget at fixed requested privacy budgets, when `--privacy_budgets` was supplied. Gaussian and Laplace policies are compared under the same public budget.
4. Audit utility versus configured mediator noise scale, with the two scale conventions identified.

The certificate-focused figures distinguish a guaranteed channel property from a measured attack outcome. The fixed-budget query figure tests how utility changes as the mediator increases its noise with query count while retaining the same ceiling. An experiment measures the utility cost and how closely a chosen audit approaches the ceiling; it does not empirically prove the universal proposition.

Uncertainty is estimated by resampling held-out model IDs within membership strata. Paired reference/alternative endpoints and all repeated queries for a model remain together. The same model resampling is used across policies for paired utility-loss intervals. These intervals are conditional on the fitted score models and calibration sets; they do not capture uncertainty from retraining those tests or rerunning calibration. Repeated queries/noise realizations do not create new independently trained models. With a small held-out ensemble, the intervals can be wide. Finite-sample estimates can exceed theoretical population ceilings; the outputs do not clip them to the bound.

This is an offline trusted experiment evaluator, not a deployed mediator access-control service. The auditor modeled by a mediated trial sees only that trial's permitted queries, public policy, and released responses. The evaluator's raw baseline results, target labels, checkpoints, and reproducibility seeds are not additional auditor observations. In particular, mediator noise seeds must not be published as part of a real release.

Treat every policy/repetition as a separate experimental audit. Releasing several sweeps, repetitions, or the baseline together to one auditor would require joint privacy accounting; a single-batch certificate does not cover that combined transcript. Saved experimental artifacts are not a claim that those releases can safely be combined.

For noise-response panels assembled from the paper's evidence folders, run:

```bash
python plot_noise_tradeoff.py --evidence_dir <evidence_folder> --output_dir <figure_folder> --T 30 --include-calibrated
```

The evidence folder contains `ga-eta0.1`, `ga-eta0.5`, `ga-eta0.9`, and the corresponding `salun-eta*` directories, each with `mediation_summary.csv`. Run once per dataset/deletion regime. The script saves twelve separate PDFs, such as `gaussian-GA-eta0.1-full.pdf`, plus a CSV for each panel. Each curve plots signed membership advantage and compliance error against actual mediator noise scale, retaining the conditional confidence intervals. `--include-calibrated` includes all ten policies per panel and uses a horizontal axis that is linear through 0.5 and logarithmic above it; omit the flag for the six baseline/scale policies. Assemble the PDFs as individual LaTeX subfigures.

## 5. Acknowledgements

Many of the code used in this repository are forked from the [code implementations](https://github.com/K1nght/Unified-Unlearning-w-Remain-Geometry) of [Huang et al.](https://arxiv.org/abs/2409.19732), as noted in our paper. We thank the authors for making their code public.
