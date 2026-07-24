import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import math


def compute_fim_weight(alpha, min_weight=0.3, max_weight=1.0, epsilon=1e-6):
    alpha = alpha.clamp_min(epsilon)

    alpha0 = alpha.sum(dim=1, keepdim=True)

    trigamma_alpha = torch.polygamma(1, alpha)
    trigamma_alpha0 = torch.polygamma(1, alpha0)
    fisher_diag = trigamma_alpha - trigamma_alpha0

    fisher_diag = fisher_diag.clamp_min(0.0)

    sample_fim = fisher_diag.mean(dim=1, keepdim=True)

    fim_min = sample_fim.min()
    fim_max = sample_fim.max()

    if fim_max - fim_min < epsilon:
        return torch.full_like(sample_fim, (min_weight + max_weight) / 2.0)

    normalized = (sample_fim - fim_min) / (fim_max - fim_min + epsilon)
    weight = min_weight + (max_weight - min_weight) * normalized

    return weight


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device, use_fim_weight=True):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)

    if use_fim_weight:
        fim_weight = compute_fim_weight(alpha)
        loss = loss * fim_weight

    return torch.mean(loss)


def get_loss(evidences, target, epoch_num, num_classes, annealing_step, device, use_fim_weight=True):
    loss_acc = 0.0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss_acc += edl_digamma_loss(
            alpha, target, epoch_num, num_classes, annealing_step, device, use_fim_weight
        )
    loss_acc = loss_acc / len(evidences)
    loss = loss_acc
    return loss