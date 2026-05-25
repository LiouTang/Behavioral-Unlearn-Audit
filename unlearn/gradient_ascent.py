import os

import torch
import torch.nn as nn

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class GradAscent(UnlearnMethod):
    def __init__(self, *args, **kwargs) -> None:
        super(GradAscent, self).__init__(*args, **kwargs)
        # params
        self.weight_decay = 5e-4
        self.lr = 2e-4
        self.epochs = int(self.eta * 10)

        self.model.to(self.device)
        self.model.train()

    def get_unlearned(self) -> nn.Module:
        rt_loader, un_loader = self.loaders["rt"], self.loaders["un"]
        val_loader = self.loaders["val"]

        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            for i, (input, target) in enumerate(un_loader):
                input, target = input.to(self.device), target.to(torch.int64).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input)
                loss = -1 * self.loss_fn(outputs, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
    
        _, rt_top1  = validate(self.model, rt_loader,   nn.CrossEntropyLoss(), self.device)
        _, un_top1  = validate(self.model, un_loader,   nn.CrossEntropyLoss(), self.device)
        _, val_top1 = validate(self.model, val_loader,  nn.CrossEntropyLoss(), self.device)
        print(f"rt acc: {rt_top1.avg:.2f}, un acc: {un_top1.avg:.2f}, val acc: {val_top1.avg:.2f}")
        return self.model