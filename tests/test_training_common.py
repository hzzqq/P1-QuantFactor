"""R7/R8：共享稳健训练辅助（src/training/common）测试。

覆盖：
- seed_everything 固定种子 → 两次前向结果可复现
- robust_init 不产生 NaN / 爆炸权重
- train_robust 在玩具数据上能收敛（loss 下降）且返回 history
"""
import numpy as np
import torch
import torch.nn as nn

from src.training import common as tc


class _TinyGRU(nn.Module):
    def __init__(self, n_in=4, hid=8):
        super().__init__()
        self.gru = nn.GRU(n_in, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def _toy_data(n=200, seq=10, n_in=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, seq, n_in)).astype(np.float32)
    y = X[:, -1, 0] * 1.5 + rng.normal(scale=0.1, size=n).astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return ds


def test_seed_everything_reproducible():
    tc.seed_everything(123)
    a = torch.randn(5, 3)
    tc.seed_everything(123)
    b = torch.randn(5, 3)
    assert torch.allclose(a, b), "相同种子应产生相同随机张量"


def test_robust_init_no_nan():
    m = _TinyGRU()
    tc.robust_init(m)
    for p in m.parameters():
        assert torch.isfinite(p).all(), "初始化权重不应含 NaN/Inf"
    # 跑一次前向，确认不溢出
    x = torch.randn(4, 10, 4)
    out = m(x)
    assert torch.isfinite(out).all(), "稳健初始化后前向不应溢出"


def test_train_robust_converges():
    torch.set_num_threads(2)
    ds = _toy_data()
    model = _TinyGRU()
    trained, hist = tc.train_robust(
        model, ds, epochs=6, batch_size=32, lr=1e-2, num_threads=2, seed=7,
    )
    assert len(hist) >= 2
    first = hist[0]["train_loss"]
    last = hist[-1]["train_loss"]
    assert last < first, "训练应使 loss 下降"
    # 训练后前向有限
    out = trained(torch.randn(3, 10, 4))
    assert torch.isfinite(out).all()


def test_train_robust_reg_term_runs():
    """M10 成本敏感正则项路径（reg_term）应能正常训练不报错。"""
    torch.set_num_threads(2)
    ds = _toy_data()
    model = _TinyGRU()
    reg = lambda p: 0.02 * p.pow(2).mean()
    trained, hist = tc.train_robust(
        model, ds, epochs=4, batch_size=32, lr=1e-2, num_threads=2, seed=9,
        reg_term=reg,
    )
    assert len(hist) >= 2
    assert torch.isfinite(trained(torch.randn(3, 10, 4))).all()
