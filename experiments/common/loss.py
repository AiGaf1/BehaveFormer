import torch
import torch.nn.functional as F
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(dim=1).sqrt()
    
    def calc_cosine(self, x1, x2):
        cosine_similarity = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
        return (1 - cosine_similarity) / 2

    def calc_manhattan(self, x1, x2):
        return (x1-x2).abs().sum(dim=1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_cosine(anchor, positive)
        distance_negative = self.calc_cosine(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()