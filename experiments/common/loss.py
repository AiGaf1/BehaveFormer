from pytorch_metric_learning.losses import SupConLoss
from torch.nn import TripletMarginLoss as TripletLoss

__all__ = ["TripletLoss", "SupConLoss"]
