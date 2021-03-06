import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, ignore_index=5, cuda=False):
        self.ignore_index = ignore_index
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())