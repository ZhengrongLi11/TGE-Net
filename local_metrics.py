# coding=utf-8
import torch

from piq import psnr, ssim, fsim


def global_eval(pred, target):
    # Clamp the values
    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)
    global_psnr = psnr(pred, target, data_range=1.0, reduction='mean')
    global_ssim = ssim(pred, target, data_range=1.0, reduction='mean')
    global_fsim = fsim(pred, target, data_range=1.0, reduction='mean', chromatic=False)
    return global_psnr, global_ssim * 100, global_fsim * 100


def local_eval(pred, target, label):
    if not torch.any(label):
        nan_tensor = torch.tensor(float('nan'), device=pred.device)
        return nan_tensor, nan_tensor, nan_tensor
    # Clamp the values
    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)
    y_indices, x_indices = torch.where(label[0, 0] > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    min_dim = 11
    _, _, H, W = pred.shape
    height = y_max - y_min + 1
    if height < min_dim:
        diff = min_dim - height
        y_min = max(0, y_min - diff // 2)
        y_max = min(H - 1, y_max + (diff - (diff // 2)))
        current_h = y_max - y_min + 1
        if current_h < min_dim:
            if y_min == 0:
                y_max = min(H - 1, y_max + (min_dim - current_h))
            elif y_max == H - 1:
                y_min = max(0, y_min - (min_dim - current_h))
    width = x_max - x_min + 1
    if width < min_dim:
        diff = min_dim - width
        x_min = max(0, x_min - diff // 2)
        x_max = min(W - 1, x_max + (diff - (diff // 2)))
        current_w = x_max - x_min + 1
        if current_w < min_dim:
            if x_min == 0:
                x_max = min(W - 1, x_max + (min_dim - current_w))
            elif x_max == W - 1:
                x_min = max(0, x_min - (min_dim - current_w))
    pred_crop = pred[..., y_min:y_max + 1, x_min:x_max + 1]
    target_crop = target[..., y_min:y_max + 1, x_min:x_max + 1]
    local_psnr = psnr(pred_crop, target_crop, data_range=1.0, reduction='mean')
    local_ssim = ssim(pred_crop, target_crop, data_range=1.0, reduction='mean')
    local_fsim = fsim(pred_crop, target_crop, data_range=1.0, reduction='mean', chromatic=False)
    return local_psnr, local_ssim * 100, local_fsim * 100