from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from active_adaptation.utils import torch_compat


@pytest.mark.parametrize("modern", [False, True])
def test_tf32_supports_old_and_new_pytorch_without_mixing_apis(modern):
    matmul = SimpleNamespace(fp32_precision="ieee") if modern else SimpleNamespace()
    cudnn = SimpleNamespace(conv=SimpleNamespace(fp32_precision="ieee"))
    fake = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
        set_float32_matmul_precision=Mock(),
    )
    with patch.object(torch_compat, "torch", fake):
        torch_compat.enable_tf32(cudnn=True)
    if modern:
        assert matmul.fp32_precision == "tf32"
        assert cudnn.conv.fp32_precision == "tf32"
        assert not hasattr(cudnn, "allow_tf32")
        fake.set_float32_matmul_precision.assert_not_called()
    else:
        fake.set_float32_matmul_precision.assert_called_once_with("high")
        assert cudnn.allow_tf32 is True
