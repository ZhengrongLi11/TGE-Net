# losses.py

import torch
import torch.nn.functional as F

def volume_computation(anchor, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))

    return res


def gram_contrastive_loss(text_feat, *image_feats, temperature=0.07, label_smoothing=0.1):
    """
    GRAM-style contrastive loss for multimodal alignment.

    Args:
        text_feat (Tensor): [B, D] — text modality as anchor
        *image_feats (Tensor): each [B, D] — e.g., t1, t2, flair
        temperature (float): softmax temperature
        label_smoothing (float): label smoothing factor

    Returns:
        loss (Tensor): scalar
    """
    B = text_feat.shape[0]
    device = text_feat.device

    # Compute volume matrix: text[i] vs (image_feats[j])
    volume = volume_computation(text_feat, *image_feats)  # [B, B]
    # print('--------')
    # print(text_feat)

    volume = volume / temperature

    # Transposed version: image set[j] vs text[i]
    volumeT = volume_computation(text_feat, *image_feats).T
    volumeT = volumeT / temperature

    targets = torch.linspace(0, B - 1, B, dtype=int, device=device)
    # targets = torch.arange(B, device=device).long()

    loss_i2t = F.cross_entropy(-volume, targets, label_smoothing=label_smoothing)   # text → images
    loss_t2i = F.cross_entropy(-volumeT, targets, label_smoothing=label_smoothing)  # images → text

    return (loss_i2t + loss_t2i) / 2