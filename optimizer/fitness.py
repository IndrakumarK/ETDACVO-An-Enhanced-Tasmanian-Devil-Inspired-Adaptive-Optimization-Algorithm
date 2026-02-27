import torch
import torch.nn.functional as F
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
    peak_signal_noise_ratio as psnr,
)

try:
    import lpips
    lpips_model = lpips.LPIPS(net='alex')
except:
    lpips_model = None


def compute_fitness(original,
                    augmented,
                    predictions,
                    targets,
                    task="classification"):

    # Paper-consistent weights
    w1 = 0.3  # SSIM
    w2 = 0.2  # PSNR
    w3 = 0.2  # LPIPS
    w4 = 0.3  # Dice

    ce_loss = F.cross_entropy(predictions, targets)

    ssim_penalty = 1 - ssim(augmented, original)

    psnr_val = psnr(augmented, original)
    psnr_penalty = torch.clamp(1 / (psnr_val + 1e-6), max=10.0)

    if lpips_model is not None:
        global lpips_model
        lpips_model = lpips_model.to(original.device)
        lpips_val = lpips_model(augmented, original).mean()
    else:
        lpips_val = torch.tensor(0.0, device=original.device)

    if task == "segmentation":
        preds = torch.argmax(predictions, dim=1)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_penalty = 1 - dice
    else:
        dice_penalty = torch.tensor(0.0, device=original.device)
        w4 = 0.0

    fitness = (
        ce_loss
        + w1 * ssim_penalty
        + w2 * psnr_penalty
        + w3 * lpips_val
        + w4 * dice_penalty
    )

    return fitness