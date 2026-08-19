# -*- coding: utf-8 -*-
"""
ppo_finetune.py — 在物理仿真环境中用 PPO 微调 MLP 策略

问题形态: 单步上下文决策 (contextual bandit 型 episode)
  状态  x : 30 维特征 (同 train_mlp)
  动作  a : t_start / total_time ∈ (0,1), 策略为对角高斯 N(mu(x), sigma^2)
  奖励  r : score(x, a) - 约束违反惩罚 (用每 case 的 oracle 归一化常数)

PPO 细节:
  - ratio = pi_new(a|x) / pi_old(a|x), clip 0.2
  - GAE 退化为单步 advantage: A = r - V(x)
  - 策略网络从 model_mlp.pt 初始化 (模仿学习热启动)
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from preheat_env import (Physics, RAW_BOUNDS, load_nav, load_oracle,
                         score, T_INIT, SOC_INIT, DT)
from train_mlp import PreheatMLP, build_features, NORM_PATH, MODEL_PATH as MLP_PATH

PPO_PATH = Path(__file__).parent / "model_ppo.pt"
SEED = 7


class Policy(nn.Module):
    """共享骨干 + mu/sigma 头, 从模仿学习模型热启动."""

    def __init__(self, mlp: PreheatMLP):
        super().__init__()
        self.backbone = mlp.backbone
        self.mu = nn.Sequential(mlp.head_t, nn.Sigmoid())
        self.log_sigma = nn.Parameter(torch.tensor([-3.0]))  # sigma≈0.05
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        z = self.backbone(x)
        mu = self.mu(z).squeeze(-1)
        sigma = torch.exp(self.log_sigma).expand_as(mu)
        return mu, sigma, self.value(z).squeeze(-1)


def reward_fn(res_row, tch_min, tch_max, e_min, e_max,
              pen_w=2.0):
    """res_row: [T_end, soc, E_heat, charge_time, ...]; 不可行则负惩罚."""
    T, soc, E, tch = res_row[0], res_row[1], res_row[2], res_row[3]
    lo, hi, smin = RAW_BOUNDS
    if lo <= T <= hi and soc >= smin and tch > 0:
        return score(tch, E, tch_min, tch_max, e_min, e_max) - 25.0  # 居中
    tv = max(0.0, lo - T, T - hi)
    sv = max(0.0, smin - soc)
    return -(pen_w * (tv * tv + 10.0 * sv * sv)) - 25.0


def main(iters: int = 300, batch: int = 512, lr_policy: float = 3e-5,
         clip: float = 0.2, epochs_ppo: int = 4, val_every: int = 10):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    nav = load_nav()
    feats = build_features(nav)
    total_time = (nav[:, :, 0] / nav[:, :, 1]).sum(axis=1) * 3600.0
    orc = load_oracle("raw")
    orc = orc[orc.feasible == 1].reset_index(drop=True)
    orc_tr = orc[orc.case_id <= 80000].reset_index(drop=True)

    norm = np.load(NORM_PATH)
    X_all = ((feats - norm["mu"]) / norm["sd"]).astype(np.float32)
    ids_all = orc.case_id.to_numpy() - 1

    mlp = PreheatMLP(feats.shape[1])
    ckpt = torch.load(MLP_PATH, map_location="cpu", weights_only=True)
    mlp.load_state_dict(ckpt["state_dict"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(mlp).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr_policy, weight_decay=1e-5)

    phys = Physics()
    rows = orc_tr.to_dict("records")
    vrows = orc[(orc.case_id > 80000) & (orc.case_id <= 90000)].to_dict("records")

    best_val, best_state = -np.inf, None
    t0 = time.time()
    for it in range(1, iters + 1):
        sel = rng.choice(len(rows), size=batch, replace=False)
        X = torch.tensor(X_all[ids_all[sel]], device=dev)
        with torch.no_grad():
            mu, sigma, _ = policy(X)
            a = torch.normal(mu, sigma).clamp(1e-4, 1 - 1e-4)

        # 物理仿真在 CPU 侧: 张量先回 CPU 再转 numpy
        acts = a.detach().cpu().numpy()
        rewards, vals = np.empty(batch), np.empty(batch)
        for j, si in enumerate(sel):
            r = rows[si]
            segs = nav[r["case_id"] - 1]
            sim = phys.sim(segs, float(acts[j]) * total_time[r["case_id"] - 1])
            if sim is None:
                rewards[j] = -50.0
            else:
                rewards[j] = reward_fn(
                    (sim["T_end"], sim["soc_end"], sim["E_heat"],
                     sim["charge_time"]),
                    r["t_charge_min"], r["t_charge_max"],
                    r["E_heat_min"], r["E_heat_max"])
        with torch.no_grad():
            _, _, v_old = policy(X)
            vals = v_old.detach().cpu().numpy()
        adv = torch.tensor(rewards - vals, dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        R = torch.tensor(rewards, dtype=torch.float32, device=dev)
        old_logp = torch.distributions.Normal(mu, sigma).log_prob(a)

        for _ in range(epochs_ppo):
            mu_n, sg_n, v_n = policy(X)
            dist = torch.distributions.Normal(mu_n, sg_n)
            logp = dist.log_prob(a)
            ratio = torch.exp(logp - old_logp)
            l_pol = -(torch.min(ratio * adv,
                                ratio.clamp(1 - clip, 1 + clip) * adv)).mean()
            l_val = nn.functional.smooth_l1_loss(v_n, R)
            ent = dist.entropy().mean()
            loss = l_pol + 0.5 * l_val - 0.003 * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if it % val_every == 0:
            # 验证: 直接通过率 + 平均奖励 (val 段)
            sample_v = vrows[:2000]
            npass, rsum, n = 0, 0.0, 0
            with torch.no_grad():
                for r in sample_v:
                    x = torch.tensor(X_all[r["case_id"] - 1][None, ...], device=dev)
                    mu_v, _, _ = policy(x)
                    t = float(mu_v[0]) * total_time[r["case_id"] - 1]
                    sim = phys.sim(nav[r["case_id"] - 1], t)
                    ok = (sim is not None and RAW_BOUNDS[0] <= sim["T_end"]
                          <= RAW_BOUNDS[1] and sim["soc_end"] >= RAW_BOUNDS[2])
                    npass += int(ok)
                    rsum += (reward_fn((sim["T_end"], sim["soc_end"],
                                        sim["E_heat"], sim["charge_time"]),
                                       r["t_charge_min"], r["t_charge_max"],
                                       r["E_heat_min"], r["E_heat_max"])
                              if sim else -50.0)
                    n += 1
            pr, mr = npass / n, rsum / n
            print(f"iter {it:4d}  train R {rewards.mean():+.3f}  "
                  f"val pass {pr * 100:.2f}%  val R {mr:+.3f}  "
                  f"sigma {float(torch.exp(policy.log_sigma)):.4f}  "
                  f"({time.time() - t0:.0f}s)")
            if mr > best_val:
                best_val, best_state = mr, copy.deepcopy(policy.state_dict())

    if best_state is not None:
        policy.load_state_dict(best_state)
    torch.save({"state_dict": policy.state_dict(),
                "in_dim": feats.shape[1]}, PPO_PATH)
    print(f"saved {PPO_PATH} (best val reward {best_val:+.3f})")


if __name__ == "__main__":
    main()
