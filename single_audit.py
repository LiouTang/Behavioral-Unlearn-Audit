import os
import csv
import datetime
import argparse
import pickle as pkl
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import IdxDataset
from model import get_model
from unlearn import get_unlearn_method, UnlearnMethod
from utils.operational_audit import (
    get_probabilities, split_model_runs, fit_lrt, lrt_scores,
    calibrate_threshold, empirical_rates,
)
from utils.mediation import build_policies, add_mediator_noise
from utils.mediation_results import summarize_results

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--base_dir",   type=str,   default=None, required=True)
    parser.add_argument("--target_idx", type=int,   default=None)
    parser.add_argument("--unlearn",    type=str,   default="GradAscent")
    parser.add_argument("--eta",        type=float, default=0.1,
                        help="Alternative schedule fraction; reference uses 1.0.")
    parser.add_argument("--T_max",      type=int,   default=30)
    parser.add_argument("--repeats",    type=int,   default=30)
    parser.add_argument("--sigma",      type=float, default=0.0,
                        help="Gaussian observation-noise standard deviation, before mediation.")
    parser.add_argument("--mediators", nargs="+", choices=["gaussian", "laplace"],
                        default=["gaussian", "laplace"])
    parser.add_argument("--noise_scales", type=float, nargs="+", default=[0.1, 0.5, 1.0],
                        help="Gaussian standard deviations or Laplace scales to sweep.")
    parser.add_argument("--privacy_budgets", type=float, nargs="*", default=[],
                        help="Additional domain-wide TV budgets in (0,1), calibrated separately at each T.")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--overwrite",  action="store_true")
    return parser.parse_args()


def get_un_model(i, loaders, orig_model, weight_path, args, eta):
    model = copy.deepcopy(orig_model)
    if os.path.exists(weight_path) and not args.overwrite:
        model.load_state_dict(torch.load(weight_path, map_location=args.device))
        return model

    model.load_state_dict(torch.load(os.path.join(args.base_dir, "orig", f"{i}.pth"), map_location=args.device))
    model.to(args.device)
    # Match the random update stream for the two schedules of each run.
    torch.manual_seed(args.seed + i)
    np.random.seed(args.seed + i)
    un: UnlearnMethod = get_unlearn_method(args.unlearn)(
        model=model,
        loaders=loaders,
        loss_fn=nn.CrossEntropyLoss(),
        eta=eta,
        device=args.device,
    )
    model_un = un.get_unlearned()
    torch.save(model_un.state_dict(), weight_path)
    return model_un


def search_target(dataset: IdxDataset, rng: np.random.Generator):
    candidates = utils.set_minus(np.arange(len(dataset.full_set)), dataset.un_idx)
    candidates = candidates[np.asarray(dataset.full_set.targets)[candidates] == 0]
    return int(rng.choice(candidates))


def membership_populations(reference, alternative, member_mask):
    # The same 50/50 update mixture is used in both membership worlds.
    return (
        np.concatenate((reference[member_mask], alternative[member_mask])),
        np.concatenate((reference[~member_mask], alternative[~member_mask])),
    )


def evaluate_transcripts(H, D, member_mask, splits):
    """Fit, calibrate and evaluate both tests using only released responses."""
    fit_idx, cal_idx, eval_idx = (splits[name] for name in ("fit", "calibration", "evaluation"))
    eval_members = member_mask[eval_idx]
    comp_test = fit_lrt(H[fit_idx], D[fit_idx])
    cur_test = fit_lrt(*membership_populations(H[fit_idx], D[fit_idx], member_mask[fit_idx]))
    comp_cal = (lrt_scores(comp_test, H[cal_idx]), lrt_scores(comp_test, D[cal_idx]))
    cur_cal = tuple(lrt_scores(cur_test, X) for X in membership_populations(
        H[cal_idx], D[cal_idx], member_mask[cal_idx],
    ))
    comp_threshold = calibrate_threshold(*comp_cal)
    cur_threshold = calibrate_threshold(*cur_cal)
    comp_cal_rates = empirical_rates(*comp_cal, comp_threshold)
    cur_cal_rates = empirical_rates(*cur_cal, cur_threshold)

    # Both frozen tests consume exactly the same realized transcript.
    comp_scores = (lrt_scores(comp_test, H[eval_idx]), lrt_scores(comp_test, D[eval_idx]))
    cur_scores = (lrt_scores(cur_test, H[eval_idx]), lrt_scores(cur_test, D[eval_idx]))
    comp = empirical_rates(*comp_scores, comp_threshold)
    cur = empirical_rates(*membership_populations(*cur_scores, eval_members), cur_threshold)
    cur_by_update = [
        empirical_rates(scores[eval_members], scores[~eval_members], cur_threshold)
        for scores in cur_scores
    ]
    alpha, alpha_prime = comp["fpr"], comp["fnr"]
    metrics = {
        "alpha": alpha, "alpha_prime": alpha_prime, "e_op": alpha + alpha_prime,
        "compliance_balanced_accuracy": comp["balanced_accuracy"],
        "beta": cur["J"], "membership_advantage": cur["J"],
        "membership_tpr": cur["tpr"], "membership_fpr": cur["fpr"],
        "membership_balanced_accuracy": cur["balanced_accuracy"],
        "membership_advantage_reference": cur_by_update[0]["J"],
        "membership_advantage_alternative": cur_by_update[1]["J"],
        "membership_tpr_reference": cur_by_update[0]["tpr"],
        "membership_fpr_reference": cur_by_update[0]["fpr"],
        "membership_tpr_alternative": cur_by_update[1]["tpr"],
        "membership_fpr_alternative": cur_by_update[1]["fpr"],
        "compliance_threshold": comp_threshold, "membership_threshold": cur_threshold,
        "calibration_alpha": comp_cal_rates["fpr"],
        "calibration_alpha_prime": comp_cal_rates["fnr"],
        "calibration_membership_advantage": cur_cal_rates["J"],
        "n_fit": len(fit_idx), "n_calibration": len(cal_idx), "n_evaluation": len(eval_idx),
        "n_eval_member": int(eval_members.sum()), "n_eval_nonmember": int((~eval_members).sum()),
    }
    predictions = []
    for update, comp_score, cur_score in zip(("reference", "alternative"), comp_scores, cur_scores):
        for j, model_idx in enumerate(eval_idx):
            predictions.append({
                "model_idx": int(model_idx), "update": update, "member": int(eval_members[j]),
                "compliance_score": comp_score[j], "membership_score": cur_score[j],
                "accept_reference": int(comp_score[j] >= comp_threshold),
                "predict_member": int(cur_score[j] >= cur_threshold),
            })
    return metrics, predictions


def main():
    args = parse_args()
    with open(os.path.join(args.base_dir, "env_train.pkl"), "rb") as f:
        env = pkl.load(f)
    dataset: IdxDataset = env["dataset"]
    for eta in (1.0, args.eta):
        os.makedirs(os.path.join(args.base_dir, f"{args.unlearn}-{eta}"), exist_ok=True)

    target_seed, split_seed, query_seed, observation_seed, mediator_seed = np.random.SeedSequence(args.seed).spawn(5)
    target_rng = np.random.default_rng(target_seed)
    target_idx = args.target_idx if args.target_idx is not None else search_target(dataset, target_rng)
    member_mask = np.asarray([
        target_idx in utils.set_minus(indices, dataset.un_idx)
        for indices in dataset.idx_across_N
    ], dtype=bool)
    splits = split_model_runs(member_mask, np.random.default_rng(split_seed))

    # Fix every Q before collecting model responses. The pool is only an
    # inference cache across separate experiments, each with its own T budget.
    query_rng = np.random.default_rng(query_seed)
    query_records = [
        {"T": T, "rep": rep,
         "query_idx": query_rng.choice(len(dataset.valid_set), T, replace=False)}
        for T in range(1, args.T_max + 1)
        for rep in range(args.repeats)
    ]
    query_pool_idx = np.unique(np.concatenate([record["query_idx"] for record in query_records]))
    query_loader = DataLoader(
        Subset(dataset.valid_set, query_pool_idx), batch_size=env["args"].batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    # The mediator publishes policies using only the public specification.
    policies_by_T = {
        T: [
            {"T": T, "observation_sigma": args.sigma, **policy}
            for policy in build_policies(T, args.sigma, args.mediators, args.noise_scales, args.privacy_budgets)
        ]
        for T in range(1, args.T_max + 1)
    }
    policy_columns = [
        "T", "policy_id", "mediator", "sweep", "sweep_value", "requested_budget",
        "noise_scale", "observation_sigma", "b_cert", "certificate_kind",
        "audit_accuracy_ceiling", "error_sum_floor",
    ]
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S-%f")
    result_dir = os.path.join(args.base_dir, f"mediated_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "policies.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=policy_columns)
        writer.writeheader()
        for policies in policies_by_T.values():
            writer.writerows(policies)
    # This is trusted evaluator metadata, not an artifact released to the auditor.
    with open(os.path.join(result_dir, "env_audit.pkl"), "wb") as f:
        pkl.dump({
            "args": args, "target_idx": target_idx, "member_mask": member_mask,
            "splits": splits, "query_pool_idx": query_pool_idx,
            "protocol": "mediated_fixed_query", "response": "full_posterior",
            "membership_update_prior": {"reference": 0.5, "alternative": 0.5},
            "calibration_objective": "minimum_balanced_error",
        }, f)
    with open(os.path.join(result_dir, "query_records.pkl"), "wb") as f:
        pkl.dump(query_records, f)

    hon_list, dis_list = [], []
    for i in range(env["args"].N):
        un_set = dataset.get_subset(dataset.un_idx)
        rt_set = dataset.get_subset(utils.set_minus(dataset.idx_across_N[i], dataset.un_idx))
        loaders = {
            "un": DataLoader(un_set, batch_size=env["args"].batch_size, shuffle=True, num_workers=0, pin_memory=True),
            "rt": DataLoader(rt_set, batch_size=env["args"].batch_size, shuffle=True, num_workers=0, pin_memory=True),
            "val": DataLoader(dataset.valid_set, batch_size=env["args"].batch_size, shuffle=False, num_workers=0, pin_memory=True),
        }
        model = get_model(env["args"].model, num_classes=dataset.num_classes)
        for eta, responses in ((1.0, hon_list), (args.eta, dis_list)):
            endpoint = get_un_model(
                i, loaders, model,
                os.path.join(args.base_dir, f"{args.unlearn}-{eta}", f"{i}.pth"), args, eta,
            )
            endpoint.to(args.device)
            responses.append(get_probabilities(endpoint, query_loader, args.device))
            endpoint.cpu()
        print(f"Collected responses for model {i}")
    hon_list, dis_list = np.stack(hon_list), np.stack(dis_list)

    metric_columns = [
        "alpha", "alpha_prime", "e_op", "compliance_balanced_accuracy",
        "beta", "membership_advantage", "membership_tpr", "membership_fpr",
        "membership_balanced_accuracy", "membership_advantage_reference",
        "membership_advantage_alternative", "membership_tpr_reference",
        "membership_fpr_reference", "membership_tpr_alternative", "membership_fpr_alternative",
        "compliance_threshold", "membership_threshold", "calibration_alpha",
        "calibration_alpha_prime", "calibration_membership_advantage",
        "n_fit", "n_calibration", "n_evaluation", "n_eval_member", "n_eval_nonmember",
    ]
    theory_columns = [
        "T", "rep", "n_mem", "n_non", "c_norm", "p_norm", "gamma",
        "p_parallel_norm", "p_perp_norm", "c_balanced_norm", "gamma_balanced",
        "interaction_norm", "interaction_ratio", "d_comp_ref", "d_cur_ref", "gamma_ref",
        "p_parallel_ref_norm", "p_perp_ref_norm", "e_ref", "beta_ref",
        "beta_lb_ref", "beta_lb_simple_ref",
    ]
    prediction_columns = [
        "T", "rep", "policy_id", "model_idx", "update", "member", "compliance_score",
        "membership_score", "accept_reference", "predict_member",
    ]
    observation_rng = np.random.default_rng(observation_seed)
    mediator_rng = np.random.default_rng(mediator_seed)
    fit_idx = splits["fit"]
    with open(os.path.join(result_dir, "audit_results.csv"), "w", newline="") as audit_file, \
         open(os.path.join(result_dir, "theory_results.csv"), "w", newline="") as theory_file, \
         open(os.path.join(result_dir, "heldout_predictions.csv"), "w", newline="") as prediction_file:
        audit_writer = csv.DictWriter(audit_file, fieldnames=policy_columns + ["rep"] + metric_columns)
        theory_writer = csv.DictWriter(theory_file, fieldnames=theory_columns)
        prediction_writer = csv.DictWriter(prediction_file, fieldnames=prediction_columns)
        audit_writer.writeheader()
        theory_writer.writeheader()
        prediction_writer.writeheader()

        for record in query_records:
            T, rep = record["T"], record["rep"]
            pool_pos = np.searchsorted(query_pool_idx, record["query_idx"])
            H_raw = hon_list[:, pool_pos].reshape(len(member_mask), -1).astype(np.float64)
            D_raw = dis_list[:, pool_pos].reshape(len(member_mask), -1).astype(np.float64)
            H_observed = H_raw + observation_rng.normal(0.0, args.sigma, H_raw.shape)
            D_observed = D_raw + observation_rng.normal(0.0, args.sigma, D_raw.shape)
            # Each policy is a separate experiment, never a joint release.
            for policy in policies_by_T[T]:
                H = add_mediator_noise(H_observed, policy, mediator_rng)
                D = add_mediator_noise(D_observed, policy, mediator_rng)
                metrics, predictions = evaluate_transcripts(H, D, member_mask, splits)
                audit_writer.writerow({**policy, "rep": rep, **metrics})
                prediction_writer.writerows(
                    {"T": T, "rep": rep, "policy_id": policy["policy_id"], **prediction}
                    for prediction in predictions
                )
                if policy["mediator"] == "none":
                    # The old fitted geometry remains a baseline diagnostic;
                    # it is not a lower bound after arbitrary mediation.
                    theory = utils.theory_reference_geometry(
                        H[fit_idx], D[fit_idx], member_mask[fit_idx], np.arange(H.shape[1]),
                    )
                    theory_writer.writerow({"T": T, "rep": rep, **theory})
                print(f"T={T}, rep={rep}, policy={policy['policy_id']}: "
                      f"accuracy={metrics['compliance_balanced_accuracy']:.4f}, b_cert={policy['b_cert']:.4f}")

    summarize_results(result_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    print(f"Saved mediated evaluation to {result_dir}")


if __name__ == "__main__":
    main()
