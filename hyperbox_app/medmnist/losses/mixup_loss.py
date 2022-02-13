import torch


class MixupLoss(torch.nn.Module):
    def __init__(self, criterion):
        super(MixupLoss, self).__init__()
        self.criterion = criterion
        self.training = False

    def forward(self, logits, y, *args, **kwargs):
        # y.shape is [bs,3], each column is [real labels, permuted labels, lam weight]
        if self.training:
            loss_a = self.criterion(logits, y[:, 0].long(), *args, **kwargs)
            loss_b = self.criterion(logits, y[:, 1].long(), *args, **kwargs)
            return (y[:, 2] * loss_a + (1 - y[:, 2]) * loss_b).mean()
        else:
            return self.criterion(logits, y, *args, **kwargs)

