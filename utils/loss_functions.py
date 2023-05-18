import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
from torch.nn import Sigmoid


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # apply activation function on final layer output
        inputs = Sigmoid()(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()  # 1 * 1 will be 1, rest zero
        dice_loss = 1. - (2. * intersection) / (inputs.sum() + targets.sum()).clamp(min=1e-6)  # + smooth
        bce_loss = binary_cross_entropy(inputs, targets)
        dice_bce_loss = dice_loss + bce_loss
        return dice_bce_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # apply activation function on final layer output
        inputs = Sigmoid()(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()  # 1 * 1 will be 1, rest zero
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2.):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        dice = DiceLoss()(inputs, targets)

        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = Sigmoid()(inputs)

        # focal loss
        eps = torch.finfo(torch.float32).eps
        inputs = torch.clip(inputs, eps, 1 - eps)
        single_cross_entropy = -targets * torch.log(inputs)
        focal_loss = torch.pow(1 - inputs, self.gamma) * single_cross_entropy
        focal_loss = torch.mean(torch.sum(focal_loss, dim=-1))

        # combination
        dice_focal_loss = dice + focal_loss
        return dice_focal_loss





