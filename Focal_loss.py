import torch
import torch.nn as nn
# --------------------------- BINARY LOSSES ---------------------------
class FocalLoss_B(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        # ~ super(FocalLoss_B, self).__init__()
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            # ~ print('labels != self.ignore = ',labels != self.ignore)
            # ~ mask = labels != self.ignore
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
# --------------------------- MULTICLASS LOSSES ---------------------------
class FocalLoss_M(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        print('preds = ',preds.shape)
        print('labels = ',labels.shape)
        print('preds = ',type(preds),'labels = ',type(labels))
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
