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
    # print(rt_idx_0, dataset.full_set.targets[rt_idx_0], "|||", np.where(np.array(dataset.full_set.targets)[rt_idx_0] == 0))
    target_idx = rng.choice(np.where(np.array(dataset.full_set.targets)[rt_idx_0] == 0)[0])
    return target_idx

def main():
    args = parse_args()
    with open(os.path.join(args.base_dir, "env_train.pkl"), "rb") as f:
        env = pkl.load(f)
    dataset: IdxDataset = env["dataset"]
    os.makedirs(os.path.join(args.base_dir, f"{args.unlearn}-{1.0:.1f}"), exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, f"{args.unlearn}-{args.eta:.1f}"), exist_ok=True)

    # select z^* to ensure rho(z^* | D_u) --> 1
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
        # return
        # dishonest unlearning
        model_un_dis = get_un_model(i, loaders, model, os.path.join(args.base_dir, f"{args.unlearn}-{args.eta:.1f}", f"{i}.pth"), args, args.eta)
        model_un_dis.to(args.device)
        logits_un_dis = utils.get_logit(model_un_dis, query_loader, args.device)

        hon_list.append(logits_un_hon)
        dis_list.append(logits_un_dis)

        if target_idx in rt_idx:
            mem_list.append(logits_un_hon)
            mem_list.append(logits_un_dis)
        else:
            non_list.append(logits_un_hon)
            non_list.append(logits_un_dis)


    hon_list = np.stack(hon_list)
    dis_list = np.stack(dis_list)
    mem_list = np.stack(mem_list)
    non_list = np.stack(non_list)

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    os.makedirs(os.path.join(args.base_dir, f"results_{timestamp}"), exist_ok=True)
    with open(os.path.join(args.base_dir, f"results_{timestamp}", f"env_audit.pkl"), "wb") as f:
        pkl.dump({"args": args, "target_idx": target_idx}, f)

    # rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.base_dir, f"results_{timestamp}", f"audit_results.csv"), "w") as f:
        f.write(f"T,alpha,alpha_prime,beta\n")
    for T in range(1, args.T_max + 1):
        for i in range(30):
            query_idx = rng.choice(len(dataset.valid_set), T, replace=False)

            comp = utils.LiR_test(dis_list[:,query_idx], hon_list[:,query_idx], cov_mode="diag")
            cur  = utils.LiR_test(mem_list[:,query_idx], non_list[:,query_idx], cov_mode="diag")

            alpha, alpha_prime, beta = comp["fnr"], comp["fpr"], cur["J"]
            with open(os.path.join(args.base_dir, f"results_{timestamp}", f"audit_results.csv"), "a") as f:
                f.write(f"{T},{alpha:.4f},{alpha_prime:.4f},{beta:.4f}\n")

            if i % 10 == 0:
                print(T, i)

if __name__ == "__main__":
    main()
