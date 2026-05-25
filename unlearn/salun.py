import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class SalUn(UnlearnMethod):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mask = self.zerolike_params_dict(self.model)

        # params
        self.weight_decay = 5e-4
        self.lr = 0.007
        self.epochs = int(self.eta * 10)
        self.th = 0.2

        self.mask = self.get_gradient_ratio(forget_loader=self.loaders["rt"])
        print(f"SalUn threshold: {self.th}")

    def get_unlearned(self) -> nn.Module:
        rt_loader, un_loader = self.loaders["rt"], self.loaders["un"]
        val_loader = self.loaders["val"]
        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(self.epochs):
            for i, (input, targets) in enumerate(rt_loader):
                input, targets = input.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input)
                loss = self.loss_fn(outputs, targets.to(torch.int64))
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad *= self.mask[name]

                optimizer.step()
            scheduler.step()

        _, rt_top1  = validate(self.model, rt_loader,     nn.CrossEntropyLoss(), self.device)
        _, un_top1  = validate(self.model, un_loader,     nn.CrossEntropyLoss(), self.device)
        _, val_top1 = validate(self.model, val_loader,    nn.CrossEntropyLoss(), self.device)
        print(f"rt acc: {rt_top1.avg:.2f}, un acc: {un_top1.avg:.2f}, val acc: {val_top1.avg:.2f}")
        return self.model

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        return dict([
            (k, torch.zeros_like(p, device=self.device))
            for k, p in model.named_parameters()
        ])

    def get_gradient_ratio(self, forget_loader):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        gradients = self.zerolike_params_dict(self.model)
        self.model.eval()
        for i, (images, target) in enumerate(forget_loader):
            images, target = images.to(self.device), target.to(self.device)

            # compute output
            output_clean = self.model(images)
            loss = - self.loss_fn(output_clean, target.to(torch.int64))

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data

        with torch.no_grad():
            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

        sorted_dict_positions = {}
        hard_dict = {}
        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        th_idx = int(len(all_elements) * self.th)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        i = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[i : i + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            th_tensor = torch.zeros_like(tensor_ranks)
            th_tensor[tensor_ranks < th_idx] = 1
            th_tensor = th_tensor.reshape(tensor.shape)
            hard_dict[key] = th_tensor
            i += num_elements

        return hard_dict