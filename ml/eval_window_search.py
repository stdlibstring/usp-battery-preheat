# -*- coding: utf-8 -*-
"""eval_window_search.py — 离线评估 "MLP 缩区间 + 窗口物理搜索" 设计

对比 4 种变体 (共享同一批仿真):
  W60-batch   : ±60s 窗口, 批内 min-max 选点 (当前 C 实现)
  W120-batch  : ±120s 窗口, 批内 min-max 选点
  W120-fixed  : ±120s 窗口, 固定 R_C/R_E (训练集中位数) 加权选点
  W120-true   : ±120s 窗口, 真实 R_C/R_E (理论上界, 仅离线可得)

口径: 与 eval_offline 一致, test = raw oracle 可行 case 且 case_id>90000
"""
from __future__ import annotations

import time

import numpy as np
import torch

from preheat_env import (Physics, RAW_BOUNDS, load_nav, load_oracle, score,
                         TRAIN_END)
from train_mlp import PreheatMLP, NORM_PATH, build_features

W_BIG = 120.0

phys = Physics()
nav = load_nav()
orc = load_oracle("raw")
orc = orc[orc.feasible == 1].reset_index(drop=True)

# 训练集范围统计 (固定选点法的常数)
tr = orc.case_id <= TRAIN_END
RC = orc.t_charge_max - orc.t_charge_min
RE = orc.E_heat_max - orc.E_heat_min
RC_MED = float(np.median(RC[tr]))
RE_MED = float(np.median(RE[tr]))
print("R_C (充电时间范围): median %.2f s, quantiles %s" %
      (RC_MED, np.percentile(RC, [5, 25, 75, 95]).round(2)))
print("R_E (能耗范围):     median %.3f kWh, quantiles %s" %
      (RE_MED, np.percentile(RE, [5, 25, 75, 95]).round(3)))
print("R_C/R_E median ratio: %.1f s/kWh\n" % (RC_MED / RE_MED))

test = orc[orc.case_id > 90000].reset_index(drop=True)
tt_all = (nav[:, :, 0] / nav[:, :, 1]).sum(axis=1) * 3600.0

ckpt = torch.load(NORM_PATH.parent / "model_mlp.pt", map_location="cpu",
                  weights_only=True)
model = PreheatMLP(ckpt["in_dim"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
feats = build_features(nav)
norm = np.load(NORM_PATH)
ids = test.case_id.to_numpy() - 1
Xn = torch.tensor((feats[ids] - norm["mu"]) / norm["sd"], dtype=torch.float32)
with torch.no_grad():
    fracs = model(Xn)[0].numpy().squeeze()

LO, HI, SMIN = RAW_BOUNDS


def fmask(res):
    return (res[:, 0] >= LO) & (res[:, 0] <= HI) & (res[:, 1] >= SMIN) & (res[:, 3] > 0)


def pick_batch(C, E):
    cr, er = C.max() - C.min(), E.max() - E.min()
    s = (np.full(len(C), 30.0) if cr <= 1e-9 else 30.0 * (C.max() - C) / cr)
    s = s + (np.full(len(C), 20.0) if er <= 1e-9 else 20.0 * (E.max() - E) / er)
    return int(np.argmax(s))


def pick_fixed(C, E):
    return int(np.argmin(30.0 * C / RC_MED + 20.0 * E / RE_MED))


variants: dict[str, dict] = {}


def rec(name, gap, sims):
    v = variants.setdefault(name, {"gap": [], "sims": []})
    v["gap"].append(gap)
    v["sims"].append(sims)


t0 = time.time()
n_fallback = 0
for i, row in test.iterrows():
    cid = int(row.case_id)
    segs = nav[cid - 1]
    tt = tt_all[cid - 1]
    tp = float(fracs[i]) * tt

    c = int(np.floor(tp))
    wl = max(0, c - int(W_BIG))
    wh = min(int(np.ceil(tt)), c + int(W_BIG))
    ts = np.arange(wl, wh + 1, 1.0)
    res = phys.sim_many(segs, ts)
    fm = fmask(res)
    n_sims = len(ts)

    rc = row.t_charge_max - row.t_charge_min
    re_ = row.E_heat_max - row.E_heat_min

    def gscore(C, E, k):
        return score(C[k], E[k], row.t_charge_min, row.t_charge_max,
                     row.E_heat_min, row.E_heat_max)

    if not fm.any():
        # 全路线兜底 (与 C 版 optimize_preheat 同构)
        tsf = np.arange(0.0, tt + 1e-9, 1.0)
        resf = phys.sim_many(segs, tsf)
        n_sims += len(tsf)
        n_fallback += 1
        idx = np.where(fmask(resf))[0]
        C, E = resf[idx, 3], resf[idx, 2]
        k = pick_batch(C, E)
        gap = row.score - gscore(C, E, k)
        for name in ("W60-batch", "W120-batch", "W120-fixed", "W120-true"):
            rec(name, gap, n_sims)
        # W60 窗口点数 (虽然本路径已全扫)
        continue

    idx = np.where(fm)[0]
    tsw = ts[idx]
    C, E = res[idx, 3], res[idx, 2]

    kb = pick_batch(C, E)
    kf = pick_fixed(C, E)
    kt = int(np.argmin(30.0 * C / rc + 20.0 * E / re_))
    rec("W120-batch", row.score - gscore(C, E, kb), n_sims)
    rec("W120-fixed", row.score - gscore(C, E, kf), n_sims)
    rec("W120-true", row.score - gscore(C, E, kt), n_sims)

    # W60 子窗口 (同一 1s 栅格)
    wl6 = max(wl, tp - 60.0)
    wh6 = min(wh, tp + 60.0)
    n60 = max(0, int(np.floor(wh6)) - int(np.ceil(wl6)) + 1)
    m60 = (tsw >= tp - 60.0) & (tsw <= tp + 60.0)
    if m60.any():
        C6, E6 = C[m60], E[m60]
        k6 = pick_batch(C6, E6)
        rec("W60-batch", row.score - gscore(C6, E6, k6), n60)
    else:
        tsf = np.arange(0.0, tt + 1e-9, 1.0)
        resf = phys.sim_many(segs, tsf)
        idx2 = np.where(fmask(resf))[0]
        C2, E2 = resf[idx2, 3], resf[idx2, 2]
        k2 = pick_batch(C2, E2)
        rec("W60-batch", row.score - gscore(C2, E2, k2), n60 + len(tsf))

    if (i + 1) % 500 == 0:
        print("%5d/%d  %.0fs" % (i + 1, len(test), time.time() - t0))

print()
print("fallback (窗口内无可行, 走全扫) case 数: %d (%.2f%%)" %
      (n_fallback, 100.0 * n_fallback / len(test)))
full_avg = float(np.mean([tt_all[c - 1] for c in test.case_id])) + 2
print("全路线 1s 全扫平均需 ~%.0f 次仿真/case\n" % full_avg)
for name in ("W60-batch", "W120-batch", "W120-fixed", "W120-true"):
    v = variants[name]
    g = np.array(v["gap"])
    s = np.array(v["sims"])
    print("%-10s 得分差 mean %.3f  P95 %.2f  P99 %.2f  max %.2f | 仿真 mean %.0f (%.1f%% 全扫)"
          % (name, g.mean(), np.percentile(g, 95), np.percentile(g, 99),
             g.max(), s.mean(), 100.0 * s.mean() / full_avg))
print("\ntotal %.0fs" % (time.time() - t0))
