"""PyTorch backend settings shared by desktop and DGX Spark environments."""

import torch


def enable_tf32(*, cudnn=False):
    """Use the PyTorch 2.9+ precision API, with a fallback for older builds."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        if cudnn:
            torch.backends.cudnn.conv.fp32_precision = "tf32"
    else:
        torch.set_float32_matmul_precision("high")
        if cudnn:
            torch.backends.cudnn.allow_tf32 = True
