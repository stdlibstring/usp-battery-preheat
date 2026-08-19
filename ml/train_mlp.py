# -*- coding: utf-8 -*-
"""
train_mlp.py — 监督学习 / 模仿学习: 用 safe_oracle 标签训练 MLP 预测预热开启时刻

输入特征 (29 维):
  24 维路段特征 (6 段 x [s_km, v_kmh, P_kW, T_env])
  + T_init, SOC_init, total_time_s, total_km, mean_T_env

输出:
  主头: t_start / total_time in [0,1] (sigmoid)
  辅助头: T_end, SOC_end (回归, 改善表征)

数据: 仅使用 oracle 可行的 case; 80k 训练 / 10k 验证 (按 case_id 划分)
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preheat_env import (T_INIT, SOC_INIT, load_nav, load_oracle, split_ids,
                         TRAIN_END, VAL_END)

MODEL_PATH = Path(__file__).parent / "model_mlp.pt"
NORM_PATH = Path(__file__).parent / "norm_mlp.npz"
SEED = 42


def build_features(nav: np.ndarray) -> np.ndarray:
    n = nav.shape[0]
    flat = nav.reshape(n, -1)                       # 24
    total_km = nav[:, :, 0].sum(axis=1)             # (n,)
    seg_time = nav[:, :, 0] / nav[:, :, 1]          # hours
    total_time_h = seg_time.sum(axis=1)
    mean_T = nav[:, :, 3].mean(axis=1)
    mean_v = nav[:, :, 1].mean(axis=1)
    extra = np.stack([
        np.full(n, T_INIT), np.full(n, SOC_INIT),
        total_time_h * 3600.0, total_km, mean_T, mean_v,
    ], axis=1)
    return np.hstack([flat, extra]).astype(np.float64)   # (n, 30)


class PreheatMLP(nn.Module):
    def __init__(self, in_dim: int = 30, hidden=(128, 128, 64)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head_t = nn.Linear(d, 1)      # -> sigmoid(t/total)
        self.head_T = nn.Linear(d, 1)      # aux: T_end
        self.head_soc = nn.Linear(d, 1)    # aux: SOC_end

    def forward(self, x):
        z = self.backbone(x)
        return (torch.sigmoid(self.head_t(z)),
                self.head_T(z),
                self.head_soc(z))


def main(epochs: int = 200, batch: int = 1024, lr: float = 1e-3,
         patience: int = 20, oracle_kind: str = "safe"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading data ...")
    nav = load_nav()
    feats = build_features(nav)
    orc = load_oracle(oracle_kind)
    orc = orc[orc.feasible == 1].reset_index(drop=True)

    total_time = (nav[:, :, 0] / nav[:, :, 1]).sum(axis=1) * 3600.0  # s

    ids = orc.case_id.to_numpy() - 1
    y_frac = (orc.best_start_time_s.to_numpy() / total_time[ids])
    y_T = orc.T_end.to_numpy()
    y_soc = orc.SOC_end_pct.to_numpy()

    tr = orc.case_id.to_numpy() <= TRAIN_END
    va = (~tr) & (orc.case_id.to_numpy() <= VAL_END)
    if va.sum() == 0:
        # 部分数据模式: 末尾 15% 作验证 (全量数据到位后自动回到标准划分)
        n_rows = len(orc)
        cut = int(n_rows * 0.85)
        tr = np.zeros(n_rows, dtype=bool)
        tr[:cut] = True
        va = ~tr
        print(f"[partial-data mode] train {int(tr.sum())} / val {int(va.sum())}")
    X = feats[ids]
    tt = total_time[ids]

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    np.savez(NORM_PATH, mu=mu, sd=sd)
    Xn = (X - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(y_frac[tr], dtype=torch.float32, device=dev).unsqueeze(1)
    ytr_T = torch.tensor(y_T[tr], dtype=torch.float32, device=dev).unsqueeze(1)
    ytr_s = torch.tensor(y_soc[tr], dtype=torch.float32, device=dev).unsqueeze(1)
    Xva = torch.tensor(Xn[va], dtype=torch.float32, device=dev)
    yva_t = y_frac[va]
    tva = tt[va]

    model = PreheatMLP(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = nn.SmoothL1Loss(beta=0.02)

    n_tr = Xtr.shape[0]
    best_mae, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = 0.0
        for i in range(0, n_tr, batch):
            idx = perm[i:i + batch]
            pt, pT, pS = model(Xtr[idx])
            loss = (huber(pt, ytr_t[idx])
                    + 0.25 * nn.functional.smooth_l1_loss(pT, ytr_T[idx], beta=0.1)
                    + 0.25 * nn.functional.smooth_l1_loss(pS, ytr_s[idx], beta=0.1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva)[0].squeeze(1).numpy()
        mae = float(np.mean(np.abs(pv - yva_t) * tva))
        if mae < best_mae - 1e-4:
            best_mae, bad = mae, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            p95 = float(np.percentile(np.abs(pv - yva_t) * tva, 95))
            print(f"ep {ep:3d}  loss {tot / n_tr:.5f}  val MAE {mae:.2f} s  "
                  f"P95 {p95:.1f} s  ({time.time() - t0:.0f}s, bad {bad})")
        if bad >= patience:
            print(f"early stop at ep {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(),
                "in_dim": X.shape[1]}, MODEL_PATH)
    print(f"saved {MODEL_PATH}; best val MAE = {best_mae:.2f} s "
          f"({oracle_kind} oracle, {int(tr.sum())} train / {int(va.sum())} val cases)")


if __name__ == "__main__":
    main()
