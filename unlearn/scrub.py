import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from .unlearn_method import UnlearnMethod
from trainer import train, validate


class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def param_dist(model, swa_model, p=0.0):
    #https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p="fro")
    return p * dist

class SCRUB(UnlearnMethod):
    def __init__(self, *args, **kwargs):
        super(SCRUB, self).__init__(*args, **kwargs)

        # params
        self.msteps = 2
        self.sstart = 10
        self.sgda_epochs = 6

        self.model_t, self.model_s = deepcopy(self.model), deepcopy(self.model)
        self.swa_model = torch.optim.swa_utils.AveragedModel(self.model_s, avg_fn=lambda x, _, __: x)

        self.module_list = nn.ModuleList([self.model_s, self.model_t])
        self.cls, self.div = nn.CrossEntropyLoss(), DistillKL(4)

    def get_unlearned(self) -> nn.Module:
        rt_loader, un_loader = self.loaders["rt"], self.loaders["un"]
        val_loader = self.loaders["val"]

        # optimizer
        optimizer = torch.optim.AdamW(self.model_s.parameters(), lr=8e-5, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.sgda_epochs)

        self.module_list.to(self.device)
        self.swa_model.to(self.device)

        for epoch in range(1, self.sgda_epochs + 1):
            if epoch <= self.msteps:
                self._train_distill(epoch, un_loader, optimizer, "max")
            self._train_distill(epoch, rt_loader, optimizer, "min")

            if epoch >= self.sstart:
                self.swa_model.update_parameters(self.model_s)

            scheduler.step(epoch)

        _, rt_top1  = validate(self.model_s, rt_loader,     nn.CrossEntropyLoss(), self.device)
        _, un_top1  = validate(self.model_s, un_loader,     nn.CrossEntropyLoss(), self.device)
        _, val_top1 = validate(self.model_s, val_loader,    nn.CrossEntropyLoss(), self.device)
        print(f"rt acc: {rt_top1.avg:.2f}, un acc: {un_top1.avg:.2f}, val acc: {val_top1.avg:.2f}")

        return self.model_s

    def _train_distill(self, epoch, train_loader, optimizer, split):
        """One epoch distillation"""
        for m in self.module_list:
            m.train()
        self.model_t.eval()

        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(self.device), target.to(torch.int64).to(self.device)

            logit_s = self.model_s(input)
            with torch.no_grad():
                logit_t = self.model_t(input)

            # cls + kl div
            loss_cls = self.cls(logit_s, target.to(torch.int64))
            loss_div = self.div(logit_s, logit_t.to(torch.int64))

            if split == "min":
                loss = 0.99 * loss_cls + 0.001 * loss_div
            elif split == "max":
                loss = -loss_div
            loss = loss + param_dist(self.model_s, self.swa_model)

            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_value_(model_s.parameters(), clip)
            optimizer.step()
        