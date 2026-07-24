# coding=utf-8
import os
import re
import cv2
import numpy as np
import torch
import random
import nibabel as nib

from options.config import load_config
from data import create_dataset
from models import create_model
from util import util

from piq import psnr as piq_psnr
from piq import ssim as piq_ssim
from piq import fsim as piq_fsim


# =========================

# =========================
def set_random_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================

# =========================
def global_eval(pred01: torch.Tensor, target01: torch.Tensor):
    pred01 = torch.clamp(pred01, 0.0, 1.0)
    target01 = torch.clamp(target01, 0.0, 1.0)
    g_psnr = piq_psnr(pred01, target01, data_range=1.0, reduction="mean")
    g_ssim = piq_ssim(pred01, target01, data_range=1.0, reduction="mean")
    g_fsim = piq_fsim(pred01, target01, data_range=1.0, reduction="mean", chromatic=False)
    return g_psnr, g_ssim * 100, g_fsim * 100


def local_eval(pred01: torch.Tensor, target01: torch.Tensor, label01: torch.Tensor):
    """
    pred01/target01: [1,1,H,W] in [0,1]
    
    """
    if not torch.any(label01):
        nan_tensor = torch.tensor(float("nan"), device=pred01.device)
        return nan_tensor, nan_tensor, nan_tensor

    pred01 = torch.clamp(pred01, 0.0, 1.0)
    target01 = torch.clamp(target01, 0.0, 1.0)

    y_indices, x_indices = torch.where(label01[0, 0] > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    min_dim = 11
    _, _, H, W = pred01.shape

    
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

    pred_crop = pred01[..., y_min:y_max + 1, x_min:x_max + 1]
    target_crop = target01[..., y_min:y_max + 1, x_min:x_max + 1]

    l_psnr = piq_psnr(pred_crop, target_crop, data_range=1.0, reduction="mean")
    l_ssim = piq_ssim(pred_crop, target_crop, data_range=1.0, reduction="mean")
    l_fsim = piq_fsim(pred_crop, target_crop, data_range=1.0, reduction="mean", chromatic=False)
    return l_psnr, l_ssim * 100, l_fsim * 100


# =========================
# 2) seg(.nii) -> 2D label
# =========================
SEG_DIR = "/data2/users/lzr/BraTS2020/AE-GAN-BraTS2020/AE-GAN-BraTS2020_test/seg"
_CASE_RE = re.compile(r"(BraTS20_(?:Training|Validation|Testing)_\d+)")

_seg_cache = {}  # case_id -> seg_vol(np.int16)


def load_seg_volume(case_id: str) -> np.ndarray:
    if case_id in _seg_cache:
        return _seg_cache[case_id]

    cand1 = os.path.join(SEG_DIR, f"{case_id}_seg.nii")
    cand2 = os.path.join(SEG_DIR, f"{case_id}_seg.nii.gz")
    seg_path = cand1 if os.path.exists(cand1) else cand2
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Seg not found: {cand1} / {cand2}")

    seg_vol = nib.load(seg_path).get_fdata().astype(np.int16)
    _seg_cache[case_id] = seg_vol
    return seg_vol


def extract_slice_2d(seg_vol: np.ndarray, slice_k: int) -> np.ndarray:
    """
    
    
    """
    if seg_vol.ndim != 3:
        raise ValueError(f"seg_vol must be 3D, got {seg_vol.shape}")

    H, W, D = seg_vol.shape
    if slice_k < D:
        return seg_vol[:, :, slice_k]

    # fallback: maybe [D,H,W]
    if slice_k < seg_vol.shape[0]:
        return seg_vol[slice_k, :, :]

    raise IndexError(f"slice_k={slice_k} out of bounds for seg shape={seg_vol.shape}")


def get_label(case_id: str, slice_k: int, out_hw=(256, 256), device="cpu") -> torch.Tensor:
    seg_vol = load_seg_volume(case_id)
    seg2d = extract_slice_2d(seg_vol, slice_k)

    
    mask = (seg2d > 0).astype(np.uint8)

    if mask.shape != out_hw:
        mask = cv2.resize(mask, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)

    label01 = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
    return label01


# =========================
# 3) tensor -> [0,1]
# =========================
def to_01(x: torch.Tensor) -> torch.Tensor:
    x_min = float(x.min().detach().cpu())
    if x_min < 0.0:
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


# =========================

# =========================
def parse_case_id_from_path(p: str):
    m = _CASE_RE.search(p)
    return m.group(1) if m else None


# =========================

# =========================
def save_mask_jpg(label01: torch.Tensor, save_path: str):
    """
    
    
    """
    mask = (label01[0, 0].detach().cpu().numpy() > 0).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, mask)


# =========================

# =========================
if __name__ == "__main__":
    set_random_seed(11)

    opt = load_config()
    opt.load_size = 256
    opt.results_dir = "results/"
    opt.num_threads = 0
    opt.batch_size = 4
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False

    phase = "val" if not opt.isTrain else "train"
    dataloader = create_dataset(opt)

    opt.n_input_modal = dataloader.dataset.n_modal - 1
    opt.modal_names = dataloader.dataset.get_modal_names()
    model = create_model(opt)
    model.setup(opt)

    dst_dir = os.path.join(opt.results_dir, opt.name, f"{phase}-{opt.epoch}")
    os.makedirs(dst_dir, exist_ok=True)

    
    mask_dir = os.path.join(dst_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    metrics = {}
    global_index = 0

    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        batch_size = next(iter(visuals.values())).size(0)

        B_paths = data.get("B_path", None)
        Slice_Index = data.get("Slice_Index", None)

        for b in range(batch_size):
            imgs = []
            labels = []

            # 1) case_id
            case_id = None
            if isinstance(B_paths, (list, tuple)) and len(B_paths) > b:
                case_id = parse_case_id_from_path(B_paths[b])
            elif isinstance(B_paths, str):
                case_id = parse_case_id_from_path(B_paths)

            
            slice_k = None
            if Slice_Index is not None:
                if torch.is_tensor(Slice_Index):
                    slice_k = int(Slice_Index[b].item()) - 1
                elif isinstance(Slice_Index, (list, tuple)):
                    slice_k = int(Slice_Index[b]) - 1
                else:
                    slice_k = int(Slice_Index) - 1

            
            if case_id is not None and slice_k is not None:
                sample_name = f"{case_id}_slice_{slice_k:03d}"
            else:
                sample_name = f"{global_index:06d}"  

            # 3) ROI label
            roi_label = None
            if case_id is not None and slice_k is not None:
                try:
                    device = next(iter(visuals.values())).device
                    roi_label = get_label(case_id, slice_k, out_hw=(opt.load_size, opt.load_size), device=device)
                except Exception as e:
                    print(f"[Warn] Failed to load seg/mask for case={case_id}, slice={slice_k}: {e}")
                    roi_label = None

            
            if roi_label is not None:
                mask_save_path = os.path.join(mask_dir, f"{sample_name}.jpg")
                save_mask_jpg(roi_label, mask_save_path)

            
            for lab, image in visuals.items():
                image_b = image[b:b+1]
                image_numpy = util.tensor2im(image_b)

                imgs.append(image_numpy)
                labels.append(lab)

                lab_dir = os.path.join(dst_dir, lab)
                os.makedirs(lab_dir, exist_ok=True)
                util.save_image(image_numpy, os.path.join(lab_dir, f"{sample_name}.jpg"))

                if lab.startswith("fake_"):
                    real_key = lab[5:]
                    if real_key in visuals:
                        if lab not in metrics:
                            metrics[lab] = {
                                "g_psnr": [], "g_ssim": [], "g_fsim": [],
                                "l_psnr": [], "l_ssim": [], "l_fsim": [],
                            }

                        fake_tensor = visuals[lab][b:b+1]
                        real_tensor = visuals[real_key][b:b+1]

                        fake01 = to_01(fake_tensor)
                        real01 = to_01(real_tensor)

                        g_psnr, g_ssim, g_fsim = global_eval(fake01, real01)
                        metrics[lab]["g_psnr"].append(g_psnr.item())
                        metrics[lab]["g_ssim"].append(g_ssim.item())
                        metrics[lab]["g_fsim"].append(g_fsim.item())

                        if roi_label is not None:
                            l_psnr, l_ssim, l_fsim = local_eval(fake01, real01, roi_label)
                            metrics[lab]["l_psnr"].append(l_psnr.item())
                            metrics[lab]["l_ssim"].append(l_ssim.item())
                            metrics[lab]["l_fsim"].append(l_fsim.item())

                            print(f"[{sample_name}] {lab} | case={case_id} slice={slice_k} "
                                  f"| G: PSNR {g_psnr.item():.2f} SSIM {g_ssim.item():.2f} FSIM {g_fsim.item():.2f} "
                                  f"| L: PSNR {l_psnr.item():.2f} SSIM {l_ssim.item():.2f} FSIM {l_fsim.item():.2f}")
                        else:
                            print(f"[{sample_name}] {lab} | case={case_id} slice={slice_k} "
                                  f"| G: PSNR {g_psnr.item():.2f} SSIM {g_ssim.item():.2f} FSIM {g_fsim.item():.2f} "
                                  f"| L: skipped(no ROI)")

            
            cat_img = np.concatenate(imgs, axis=1)
            cat_dir = os.path.join(dst_dir, "-".join(labels))
            os.makedirs(cat_dir, exist_ok=True)
            util.save_image(cat_img, os.path.join(cat_dir, f"{sample_name}.jpg"))

            global_index += 1

    
    def print_stats(name, arr):
        arr = np.array(arr, dtype=np.float64)
        if arr.size == 0:
            print(f"{name}: empty")
            return
        print(f"{name}: mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}, n={np.sum(~np.isnan(arr))}")

    for k, v in metrics.items():
        print(f"\n===== {k} =====")
        print_stats("Global PSNR", v["g_psnr"])
        print_stats("Global SSIM", v["g_ssim"])
        print_stats("Global FSIM", v["g_fsim"])
        print_stats("Local  PSNR", v["l_psnr"])
        print_stats("Local  SSIM", v["l_ssim"])
        print_stats("Local  FSIM", v["l_fsim"])

    print(f"\n[Mask Saved To] {mask_dir}")