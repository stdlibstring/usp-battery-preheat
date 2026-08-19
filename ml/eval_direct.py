# -*- coding: utf-8 -*-
"""eval_direct.py - 纯直出评估 (无安全层/无局部搜索/无兜底)

对 test 集 oracle 可行 case 抽样, 分别评估 MLP 与 PPO **直接输出**预测时刻的:
  - raw 直通率 (预测点自身满足约束)
  - 开启时刻 MAE (全体口径)
  - 可行子集得分差 vs oracle
  - 全体得分差 (不可行 case 按 0 分计, 与物理方法 100% 可行公平对比)

结果写 ml/plots/direct_eval.json 供叙事文档/图表共用。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preheat_env import (Physics, RAW_BOUNDS, load_nav, load_oracle,
                         is_feasible, score)
from train_mlp import build_features, NORM_PATH
from eval_offline import load_model

HERE = Path(__file__).parent
OUT = HERE / "plots" / "direct_eval.json"
N_SAMPLE = 2000
SEED = 20260819


def eval_direct(use_ppo: bool, nav, orc, feats, phys, total_time):
    norm = np.load(NORM_PATH)
    Xn = (feats - norm["mu"]) / norm["sd"]
    model = load_model(use_ppo)
    name = "PPO" if use_ppo else "MLP"
    with torch.no_grad():
        pred = model(torch.tensor(Xn[orc.case_id.to_numpy() - 1],
                                  dtype=torch.float32))[0].squeeze(1).numpy()

    n = len(orc)
    direct_pass = 0
    gaps_all, gaps_feas, t_mae = [], [], []
    t0 = time.time()
    for k, (_, row) in enumerate(orc.iterrows()):
        cid = int(row.case_id)
        segs = nav[cid - 1]
        tt = total_time[cid - 1]
        t_pred = float(np.clip(pred[k], 0.0, 1.0) * tt)

        t_mae.append(abs(t_pred - row.best_start_time_s))
        r = phys.sim(segs, t_pred)
        if r is not None and is_feasible(r["T_end"], r["soc_end"], RAW_BOUNDS):
            direct_pass += 1
            s = score(r["charge_time"], r["E_heat"],
                      row.t_charge_min, row.t_charge_max,
                      row.E_heat_min, row.E_heat_max)
            gaps_feas.append(row.score - s)
            gaps_all.append(row.score - s)
        else:
            gaps_all.append(float(row.score))  # 不可行 = 0 分
    dt_ = time.time() - t0
    gaps_all = np.asarray(gaps_all)
    gaps_feas = np.asarray(gaps_feas)
    return {
        "model": name,
        "n": n,
        "direct_pass_rate": direct_pass / n * 100.0,
        "t_mae_mean": float(np.mean(t_mae)),
        "t_mae_p95": float(np.percentile(t_mae, 95)),
        "gap_feas_mean": float(gaps_feas.mean()) if gaps_feas.size else None,
        "gap_feas_p95": float(np.percentile(gaps_feas, 95)) if gaps_feas.size else None,
        "gap_all_mean": float(gaps_all.mean()),
        "gap_all_p95": float(np.percentile(gaps_all, 95)),
        "sec_per_case": dt_ / n,
    }


def main():
    nav = load_nav()
    orc = load_oracle("raw")
    orc = orc[orc.feasible == 1]
    orc = orc[orc.case_id > 90000]
    rng = np.random.default_rng(SEED)
    if len(orc) > N_SAMPLE:
        orc = orc.iloc[np.sort(rng.choice(len(orc), N_SAMPLE, replace=False))]
    orc = orc.reset_index(drop=True)
    feats = build_features(nav)
    phys = Physics()
    total_time = (nav[:, :, 0] / nav[:, :, 1]).sum(axis=1) * 3600.0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for use_ppo in (False, True):
        r = eval_direct(use_ppo, nav, orc, feats, phys, total_time)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
