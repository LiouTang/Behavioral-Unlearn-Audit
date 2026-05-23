import numpy as np
import torch

import utils

def train(model, loader, loss_fn, optimizer, device, max_grad_norm=1.0) -> tuple[utils.AverageMeter, utils.AverageMeter]:
    model.train()

    losses, top1 = utils.AverageMeter(), utils.AverageMeter()
    for i, (input, target) in enumerate(loader):
        input, target = input.to(device), target.to(torch.int64).to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        top1.update(utils.accuracy(output.data, target)[0].item(), input.size(0))

    return losses, top1
