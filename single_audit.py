import os
import datetime
import argparse
import pickle as pkl
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import IdxDataset
from model import get_model
from unlearn import get_unlearn_method, UnlearnMethod

import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     type=str,   default="cuda")
    parser.add_argument("--base_dir",   type=str,   default=None, required=True)

    parser.add_argument("--target_idx", type=int,   default=None)
    parser.add_argument("--unlearn",    type=str,   default="GradAscent")
    parser.add_argument("--eta",        type=float, default=0.1)
    parser.add_argument("--T_max",      type=int,   default=50)

    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--overwrite",  action="store_true")
    return parser.parse_args()

def get_un_model(i, loaders, orig_model, weight_path, args, eta):
    model = copy.deepcopy(orig_model)
    if os.path.exists(weight_path) and not args.overwrite:
        model.load_state_dict(torch.load(weight_path))
        return model
    else:
        model.load_state_dict(torch.load(os.path.join(args.base_dir, "orig", f"{i}.pth")))
        model.to(args.device)

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
    rt_idx_0 = utils.set_minus(dataset.idx_across_N[0], dataset.un_idx)
    target_idx = rng.choice(np.where(np.array(dataset.full_set.targets)[rt_idx_0] == 0)[0])
    return target_idx

def main():
    args = parse_args()
    with open(os.path.join(args.base_dir, "env_train.pkl"), "rb") as f:
        env = pkl.load(f)
    dataset: IdxDataset = env["dataset"]
    os.makedirs(os.path.join(args.base_dir, f"{args.unlearn}-{1.0:.1f}"), exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, f"{args.unlearn}-{args.eta:.1f}"), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    target_idx = args.target_idx if args.target_idx is not None else search_target(dataset, rng)

    # full range for possible T queries, collect all logits first and cut later to save space
    query_loader = DataLoader(dataset.valid_set, batch_size=env["args"].batch_size, shuffle=False, num_workers=0, pin_memory=True)

    hon_list, dis_list, mem_list, non_list = [], [], [], []
    # run honest and dishonest unlearning
    for i in range(env["args"].N):
        un_set = dataset.get_subset(dataset.un_idx)
        rt_idx = utils.set_minus(dataset.idx_across_N[i], dataset.un_idx)
        rt_set = dataset.get_subset(rt_idx)
        loaders = {
            "un": DataLoader(un_set, batch_size=env["args"].batch_size, shuffle=True, num_workers=0, pin_memory=True),
            "rt": DataLoader(rt_set, batch_size=env["args"].batch_size, shuffle=True, num_workers=0, pin_memory=True),
            "val": DataLoader(dataset.valid_set, batch_size=env["args"].batch_size, shuffle=False, num_workers=0, pin_memory=True),
        }
        model = get_model(env["args"].model, num_classes=dataset.num_classes)
        print(f"Loaded model {i}")

        # honest unlearning
        model_un_hon = get_un_model(i, loaders, model, os.path.join(args.base_dir, f"{args.unlearn}-{1.0:.1f}", f"{i}.pth"), args, 1.0)
        model_un_hon.to(args.device)
        logits_un_hon = utils.get_logit(model_un_hon, query_loader, args.device)
        # dishonest unlearning
        model_un_dis = get_un_model(i, loaders, model, os.path.join(args.base_dir, f"{args.unlearn}-{args.eta:.1f}", f"{i}.pth"), args, args.eta)
        model_un_dis.to(args.device)
        logits_un_dis = utils.get_logit(model_un_dis, query_loader, args.device)

        hon_list.append(logits_un_hon)
        dis_list.append(logits_un_dis)

        is_member = target_idx in rt_idx
        member_mask.append(is_member)

        # Preserve the original operational MIA construction exactly:
        # both honest and dishonest endpoints enter the membership population.
        if is_member:
            mem_list.append(logits_un_hon)
            mem_list.append(logits_un_dis)
        else:
            non_list.append(logits_un_hon)
            non_list.append(logits_un_dis)


    hon_list = np.stack(hon_list)
    dis_list = np.stack(dis_list)
    mem_list = np.stack(mem_list)
    non_list = np.stack(non_list)
    member_mask = np.asarray(member_mask, dtype=bool)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    result_dir = os.path.join(args.base_dir, f"results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "env_audit.pkl"), "wb") as f:
        pkl.dump({"args": args, "target_idx": target_idx, "member_mask": member_mask}, f)

    # Preserve the existing operational output format.
    audit_path = os.path.join(result_dir, "audit_results.csv")
    with open(audit_path, "w") as f:
        f.write("T,alpha,alpha_prime,beta\n")

    # New theory/reference output. The operational CSV is intentionally
    # untouched so existing plotting code continues to work.
    theory_columns = [
        "T",
        "rep",
        "alpha",
        "alpha_prime",
        "e_op",
        "comp_tv_op",
        "beta_op",
        "n_mem",
        "n_non",
        "c_norm",
        "p_norm",
        "gamma",
        "p_parallel_norm",
        "p_perp_norm",
        "c_balanced_norm",
        "gamma_balanced",
        "interaction_norm",
        "interaction_ratio",
        "d_comp_ref",
        "d_cur_ref",
        "gamma_ref",
        "p_parallel_ref_norm",
        "p_perp_ref_norm",
        "e_ref",
        "beta_ref",
        "beta_lb_ref",
        "beta_lb_simple_ref",
    ]

    theory_path = os.path.join(result_dir, "theory_results.csv")
    with open(theory_path, "w") as f:
        f.write(",".join(theory_columns) + "\n")

    # Save the sampled query sequences as well, so every theory/operational
    # point can be reproduced or inspected later without changing the CSV.
    query_records = []

    for T in range(1, args.T_max + 1):
        for i in range(30):
            query_idx = rng.choice(len(dataset.valid_set), T, replace=False)

            comp = utils.LiR_test(dis_list[:,query_idx], hon_list[:,query_idx], cov_mode="diag")
            cur  = utils.LiR_test(mem_list[:,query_idx], non_list[:,query_idx], cov_mode="diag")

            alpha, alpha_prime, beta = comp["fnr"], comp["fpr"], cur["J"]
            e_op = alpha + alpha_prime
            comp_tv_op = comp["J"]

            with open(audit_path, "a") as f:
                f.write(f"{T},{alpha:.4f},{alpha_prime:.4f},{beta:.4f}\n")

            # New inference/post-processing-only theory-side calculation.
            theory = utils.theory_reference_geometry(
                hon_list=hon_list,
                dis_list=dis_list,
                member_mask=member_mask,
                query_idx=query_idx,
            )

            row = {
                "T": T,
                "rep": i,
                "alpha": alpha,
                "alpha_prime": alpha_prime,
                "e_op": e_op,
                "comp_tv_op": comp_tv_op,
                "beta_op": beta,
                **theory,
            }

            with open(theory_path, "a") as f:
                f.write(",".join(str(row[col]) for col in theory_columns) + "\n")

            query_records.append(
                {
                    "T": T,
                    "rep": i,
                    "query_idx": np.asarray(query_idx, dtype=np.int64),
                }
            )

            if i % 10 == 0:
                print(T, i)

    with open(os.path.join(result_dir, "query_records.pkl"), "wb") as f:
        pkl.dump(query_records, f)


if __name__ == "__main__":
    main()
