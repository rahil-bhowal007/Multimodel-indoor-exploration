import torch
import torch.nn.functional as F


def binary_dilation(binary_img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform binary dilation on a batch of masks.

    Args:
        binary_img: Tensor of shape (B, 1, H, W) with 0/1 values.
        kernel: Tensor of shape (1, 1, kH, kW) with 0/1 structuring element.

    Returns:
        Dilated tensor of same shape as binary_img.
    """
    padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
    conv = F.conv2d(binary_img, kernel, padding=padding)
    return (conv > 0).float()


def binary_erosion(binary_img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform binary erosion by dilating the inverted mask.
    """
    inv = 1 - binary_img
    dilated_inv = binary_dilation(inv, kernel)
    return 1 - dilated_inv


def binary_opening(binary_img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Opening = erosion followed by dilation.
    """
    eroded = binary_erosion(binary_img, kernel)
    return binary_dilation(eroded, kernel)


def binary_closing(binary_img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Closing = dilation followed by erosion.
    """
    dilated = binary_dilation(binary_img, kernel)
    return binary_erosion(dilated, kernel)


def binary_denoise(binary_img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Perform opening then closing to remove small artifacts.
    """
    opened = binary_opening(binary_img, kernel)
    return binary_closing(opened, kernel)
