from pytorch_metric_learning import losses


def triplet_loss(margin=0.2):
    return losses.TripletMarginLoss(margin=margin)


def supcon_loss(temperature=0.07):
    return losses.SupConLoss(temperature=temperature)


def infonce_loss(temperature=0.07):
    return losses.NTXentLoss(temperature=temperature)
