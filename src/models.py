"""예측 모델 3종(GRU·LSTM·TCN). 과거 L칸 + 미래 12칸의 알려진 정보 → 12칸 × 2변수 변화량."""
import torch
from torch import nn

H, NT = 12, 2


class Head(nn.Module):
    def __init__(self, hid, fut_dim=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hid + H * fut_dim, 128), nn.ReLU(), nn.Linear(128, H * NT))

    def forward(self, h, fut):
        return self.net(torch.cat([h, fut.flatten(1)], 1)).view(-1, H, NT)


class RNN(nn.Module):
    def __init__(self, n_in, kind="gru", hid=64, layers=2):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(n_in, hid, layers, batch_first=True, dropout=0.1)
        self.head = Head(hid)

    def forward(self, x, fut):
        out, _ = self.rnn(x)
        return self.head(out[:, -1], fut)


class TCN(nn.Module):
    """인과(causal) 팽창 합성곱. 커널 3, 팽창 1·2·4·8·16 → 수용 범위 63칸(5시간 15분)."""

    def __init__(self, n_in, ch=64, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.ModuleDict(dict(
                pad=nn.ConstantPad1d((2 * d, 0), 0.0),
                conv=nn.Conv1d(ch, ch, 3, dilation=d), act=nn.ReLU(), drop=nn.Dropout(0.1))))
        self.head = Head(ch)

    def forward(self, x, fut):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b["drop"](b["act"](b["conv"](b["pad"](h))))
        return self.head(h[:, :, -1], fut)


class AE(nn.Module):
    """정상 창(1시간 × CO₂·온도·습도)을 재구성하는 오토인코더. 시각·일정·공간은 조건으로만 넣는다."""

    def __init__(self, n_in, n_cond):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in + n_cond, 64), nn.ReLU(), nn.Linear(64, 8))
        self.dec = nn.Sequential(nn.Linear(8 + n_cond, 64), nn.ReLU(), nn.Linear(64, n_in))

    def forward(self, x, c):
        return self.dec(torch.cat([self.enc(torch.cat([x, c], 1)), c], 1))


def make(kind, n_in):
    return TCN(n_in) if kind == "tcn" else RNN(n_in, kind)
