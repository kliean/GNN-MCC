from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import losses, distances, miners


def SupervisedContrastiveLoss(x, labels):
    # SupervisedContrastiveLoss
    # distance = distances.CosineSimilarity()
    # loss_miner = miners.MultiSimilarityMiner(distance=distance)
    # loss_func = losses.SupConLoss(temperature=1.5, distance=distance)

    # tripletloss
    # distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
    # loss_miner = miners.TripletMarginMiner(margin=0.8, distance=distance, type_of_triplets="hard")
    # loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, swap=True, smooth_loss=True)

    # ContrastiveLoss
    distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
    loss_miner = miners.PairMarginMiner(pos_margin=0.8, neg_margin=1.0, distance=distance)
    loss_func = losses.ContrastiveLoss(distance=distance, pos_margin=0.1, neg_margin=1.)

    # MultiSimilarityLoss
    # distance = distances.CosineSimilarity()
    # loss_miner = miners.MultiSimilarityMiner(distance=distance)
    # loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=distance)

    loss = loss_func(x, labels)
    return loss

