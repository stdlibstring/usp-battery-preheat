# -*- coding: utf-8 -*-
"""
eval_offline.py — 离线对比评测

流程: 模型预测 -> 物理仿真检查 (raw 约束) -> ±30 s 局部 1 s 搜索
      -> 必要时全路线回退搜索

指标 (对 oracle 可行的 test case):
  - raw 直通率 / 安全层后通过率 (目标 100%)
  - 得分差 vs oracle best (均值 <= 0.5/50, P95 <= 1)
  - 开启时刻 MAE / 剩余距离 MAE
  - 平均物理仿真次数, 回退比例, 单 case 推理+校验耗时
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from preheat_env import (Physics, RAW_BOUNDS, load_nav, load_oracle,
                         is_feasible, safety_layer, score, DT)
from train_mlp import PreheatMLP, build_features, NORM_PATH

MODEL_PATH = Path(__file__).parent / "model_mlp.pt"


def load_model(use_ppo: bool = False):
    """加载 MLP 或 PPO 策略 (统一为输入特征 -> [pred (B,1)] 的接口)."""
    if not use_ppo:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model = PreheatMLP(ckpt["in_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    from ppo_finetune import Policy, PPO_PATH
    ckpt = torch.load(PPO_PATH, map_location="cpu", weights_only=True)
    policy = Policy(PreheatMLP(ckpt["in_dim"]))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    class _Wrap(torch.nn.Module):
        def forward(self, x):
            mu, _, _ = policy(x)
            return (mu.unsqueeze(1),)
    return _Wrap()


def main(n_limit: int = 0, use_ppo: bool = False):
    nav = load_nav()
    orc = load_oracle("raw")
    orc = orc[orc.feasible == 1]
    test = orc[orc.case_id > 90000]
    if len(test) == 0:
        # 部分数据模式: 用已有数据末尾 10% 作测试
        test = orc.tail(max(200, len(orc) // 10))
        print(f"[partial-data mode] test = last {len(test)} available cases")
    orc = test
    if n_limit > 0:
        orc = orc.head(n_limit)
    feats = build_features(nav)
    norm = np.load(NORM_PATH)
    Xn = (feats - norm["mu"]) / norm["sd"]
    model = load_model(use_ppo)
    print(f"model                 : {'PPO policy (model_ppo.pt)' if use_ppo else 'MLP (model_mlp.pt)'}")
    phys = Physics()

    total_time = (nav[:, :, 0] / nav[:, :, 1]).sum(axis=1) * 3600.0

    n = len(orc)
    direct_pass = 0
    final_pass = 0
    gaps, t_mae, d_mae, sims, modes = [], [], [], [], {"direct": 0, "local": 0,
                                                       "fallback": 0, "none": 0}
    t0 = time.time()
    with torch.no_grad():
        pred = model(torch.tensor(Xn[orc.case_id.to_numpy() - 1],
                                  dtype=torch.float32))[0].squeeze(1).numpy()

    for k, (_, row) in enumerate(orc.iterrows()):
        cid = int(row.case_id)
        segs = nav[cid - 1]
        tt = total_time[cid - 1]
        t_pred = float(np.clip(pred[k], 0.0, 1.0) * tt)

        r = phys.sim(segs, t_pred)
        n_sim = 1
        if r is not None and is_feasible(r["T_end"], r["soc_end"], RAW_BOUNDS):
            direct_pass += 1
            t_final, mode = t_pred, "direct"
        else:
            t_final, mode = safety_layer(phys, segs, t_pred, tt, RAW_BOUNDS)
            n_sim += (61 if mode in ("local", "none") else
                      (61 + int(tt / 10.0) + 31))
        modes[mode] += 1
        r = phys.sim(segs, t_final)
        ok = r is not None and is_feasible(r["T_end"], r["soc_end"], RAW_BOUNDS)
        if ok:
            final_pass += 1
            s = score(r["charge_time"], r["E_heat"],
                      row.t_charge_min, row.t_charge_max,
                      row.E_heat_min, row.E_heat_max)
            gaps.append(row.score - s)
            t_mae.append(abs(t_final - row.best_start_time_s))
            d_mae.append(abs(r["dist"] - row.best_start_distance_km))
        sims.append(n_sim)

    dt_ = time.time() - t0
    gaps = np.asarray(gaps)
    print(f"test cases            : {n} (oracle feasible)")
    print(f"raw direct pass rate  : {direct_pass / n * 100:.2f}%")
    print(f"final pass rate       : {final_pass / n * 100:.2f}%")
    if gaps.size:
        print(f"score gap mean        : {gaps.mean():.4f} / 50 "
              f"(max {gaps.max():.4f})")
        print(f"score gap P95         : {np.percentile(gaps, 95):.4f}")
    print(f"start-time MAE        : {np.mean(t_mae):.2f} s "
          f"(P95 {np.percentile(t_mae, 95):.1f} s)")
    print(f"distance MAE          : {np.mean(d_mae):.3f} km")
    print(f"modes                 : {modes}")
    print(f"avg sims / case       : {np.mean(sims):.1f}")
    print(f"throughput            : {n / dt_:.1f} cases/s "
          f"({dt_ * 1000 / n:.1f} ms/case, incl. physics checks)")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    use_ppo = "ppo" in args
    args = [a for a in args if a != "ppo"]
    main(int(args[0]) if args else 0, use_ppo)
