import os
import argparse
import pickle as pkl

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataset
from model import get_model
from trainer import train, validate

import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",         type=str,   default="cuda")

    parser.add_argument("--dataset",        type=str,   default="CIFAR10")
    parser.add_argument("--data_dir",       type=str,   default="./data")

    parser.add_argument("--model",          type=str,   default="ResNet18")
    parser.add_argument("--N",              type=int,   default=32)
    parser.add_argument("--un_size",        type=int,   default=1000)

    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--num_epochs",     type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)

    parser.add_argument("--base_dir",       type=str,   default="./save")
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--overwrite",      action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.base_dir, exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, "orig"), exist_ok=True)

    dataset = get_dataset(args.dataset, root=args.data_dir, seed=args.seed)
    idx_across_N, _ = dataset.train_unlearn_split(args.N, args.un_size) # pre-split the dataset

    with open(os.path.join(args.base_dir, "env_train.pkl"), "wb") as f:
        pkl.dump({
            "args": args,
            "dataset": dataset,
        }, f)

    for i in range(args.N):
        if os.path.exists(os.path.join(args.base_dir, "orig", f"{i}.pth")) and not args.overwrite:
            print(f"Skip {i}")
            continue

        train_set = dataset.get_subset(idx_across_N[i])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        model = get_model(args.model, num_classes=dataset.num_classes)
        model.to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_top1 = 0.0
        for epoch in (pbar := tqdm(range(args.num_epochs), desc=str(i).zfill(3))):
            _, _ = train(model, train_loader, loss_fn, optimizer, args.device)
            losses, top1 = validate(model, train_loader, loss_fn, args.device)
            pbar.set_postfix({"loss": losses.avg, "top1": top1.avg})
            if top1.avg > best_top1:
                best_top1 = top1.avg
                torch.save(model.state_dict(), os.path.join(args.base_dir, "orig", f"{i}.pth"))


if __name__ == "__main__":
    main()
