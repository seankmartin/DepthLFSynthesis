"""Contains the full model, including
Model
Optimizer
LR scheduler
Criterion
"""
from time import time
import torch.nn as nn
import torch.optim as optim

from model_2d import C2D
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim.lr_scheduler import CyclicLR

def setup_model(args):
    """Returns a tuple of the model, criterion, optimizer and lr_scheduler"""
    print("Building model")
    stime = time()
    model = C2D(args, inchannels=1, outchannels=1)
    criterion = nn.MSELoss(size_average=True)
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    # See https://arxiv.org/abs/1608.03983
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max = (args.nEpochs // 10) + 1)
    """lr_scheduler = CyclicLR(
        optimizer)"""
    print("Finished building model in {:2f}".format(
        time() - stime
    ))
    return (model, criterion, optimizer, lr_scheduler)