import numpy as np
import torch
import utils


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple[utils.AverageMeter, utils.AverageMeter]:
    model.eval()

    losses, top1 = utils.AverageMeter(), utils.AverageMeter()
    for i, (input, target) in enumerate(loader):
        model.eval()
        input, target = input.to(device), target.to(torch.int64).to(device)

        with torch.no_grad():
            output = model(input)
        loss = loss_fn(output, target)

        losses.update(loss.item(), input.size(0))
        top1.update(utils.accuracy(output.data, target)[0].item(), input.size(0))

    return losses, top1