import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.gather(0, target)
            focal_loss = focal_loss * alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Example usage:
# Assuming you have a model, criterion = FocalLoss(), and data
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Inside your training loop:
# output = model(input)
# loss = criterion(output, target)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
