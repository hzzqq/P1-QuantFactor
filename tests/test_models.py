"""N6 TCN 因果卷积 / N13 GRU 输入归一测试。"""
import numpy as np
import torch

from src.models.gru_attn import GRUAttention, TCN, TCNBlock


def test_tcn_is_causal_no_future_leak(N6=True):
    """N6：TCN 改为因果（仅左填充）后，改变未来时间步 t_f 的输入不得影响 t<t_f 的输出。

    直接测 TCNBlock（返回完整序列），其输入约定为 (B, C, T)。"""
    torch.manual_seed(0)
    C, T = 3, 12
    base = torch.randn(1, C, T)
    base2 = base.clone()
    base2[:, :, 8] += 100.0          # 仅在「未来」时间步 8 注入扰动
    block = TCNBlock(in_ch=C, out_ch=4, kernel=3, dilation=1)
    block.eval()   # 关闭 Dropout/BatchNorm 训练态，保证因果性对比确定性
    out1 = block(base)
    out2 = block(base2)
    # 因果性：输出时间步 t<8 必须完全不受 t=8 扰动影响（仅左填充保证）
    assert torch.allclose(out1[:, :, :8], out2[:, :, :8], atol=1e-6), \
        "TCN 仍窥见未来（前序输出随未来扰动变化）"


def test_gru_input_norm_shape(N13=True):
    """N13：GRU 输入加 LayerNorm 后 forward 形状正确且模块存在。"""
    torch.manual_seed(0)
    m = GRUAttention(n_features=4, hidden=8, n_layers=1)
    assert hasattr(m, "input_norm"), "GRUAttention 应含 input_norm"
    x = torch.randn(2, 5, 4)          # (B, T, F)
    y = m(x)
    assert y.shape == (2,), f"GRU 输出形状应为 (B,)，实得 {tuple(y.shape)}"
    # 注意力权重返回
    y2, attn = m(x, return_attn=True)
    assert y2.shape == (2,) and attn.shape == (2, 5)
