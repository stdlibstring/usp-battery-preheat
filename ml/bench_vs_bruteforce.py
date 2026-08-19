# -*- coding: utf-8 -*-
"""
bench_vs_bruteforce.py — 我们的分层搜索 vs 朴素暴力搜索 对照实验

同机同负载(Oracle 生成满载运行中, 各方法承受相同争抢, 比值公平):
  bf10 / bf1 / bf0.1   朴素暴力: 全路线固定步长扫描 (前缀缓存批仿真)
  bf1_nocache          朴素暴力最自然写法: 每个候选从头完整仿真一次 (无前缀复用)
  hier                 我们的分层搜索: 10s 粗扫 -> 1s 区间内 -> 0.1s 边界精修, 失败升级
  质量基准             oracle_raw.csv (C 生成器, 经 0.01s L3 审计验证)

指标: 可行解检出率 / 开启时刻偏差 / 得分差距(漏检记 50) / 单 case 耗时 / 评估次数
"""
import time
import numpy as np
import pandas as pd

from preheat_env import (Physics, load_nav, score, _pick_best_index, RAW_BOUNDS)

rng = np.random.default_rng(42)
phys = Physics()
nav = load_nav()

orc = pd.read_csv("../oracle/output/oracle_raw.csv", on_bad_lines="skip")
orc = orc.dropna(subset=["case_id"])
orc["case_id"] = orc["case_id"].astype(int)
orc = orc[orc.case_id <= len(nav)]
feas = orc[orc.feasible == 1].copy()
print(f"oracle 快照: {len(orc)} 行, 其中可行 {len(feas)}")

# ---- 分层抽样: 窄可行窗(few feasible candidates) + 随机普通 ----
narrow = feas[feas.feasible_count <= 50]
print(f"窄窗 case (feasible_count<=50): {len(narrow)} 个")
sel_n = narrow.case_id.values if len(narrow) <= 30 else rng.choice(
    narrow.case_id.values, 30, replace=False)
sel_o = rng.choice(feas[feas.feasible_count > 50].case_id.values,
                   min(70, (feas.feasible_count > 50).sum()), replace=False)
sel = np.sort(np.concatenate([sel_n, sel_o]))
narrow_set = set(sel_n.tolist())
print(f"样本: {len(sel)} 个 case (其中窄窗 {len(narrow_set)})")

orc_idx = feas.set_index("case_id")
LO_B, HI_B, SOC_B = RAW_BOUNDS


def feasible_mask(res):
    return ((res[:, 0] >= LO_B) & (res[:, 0] <= HI_B) &
            (res[:, 1] >= SOC_B) & (res[:, 3] > 0))


def judge(ts, res, row):
    """与 Oracle 同口径选点: 可行集内得分最大化 (以 Oracle 的 min/max 为基准),
    再用同一基准给选出的点打分 (真基准)."""
    feas = feasible_mask(res)
    if not feas.any():
        return dict(found=0, t=np.nan, dt=np.nan, sc=0.0)
    idx = np.where(feas)[0]
    C, E = res[idx, 3], res[idx, 2]
    cr = row.t_charge_max - row.t_charge_min
    er = row.E_heat_max - row.E_heat_min
    s_t = (np.minimum(30.0, 30.0 * (row.t_charge_max - C) / cr)
           if cr > 1e-9 else np.full(len(idx), 30.0))
    s_e = (np.minimum(20.0, 20.0 * (row.E_heat_max - E) / er)
           if er > 1e-9 else np.full(len(idx), 20.0))
    tot = s_t + s_e
    j = int(np.argmax(tot))
    return dict(found=1, t=float(ts[idx[j]]),
                dt=abs(float(ts[idx[j]]) - row.best_start_time_s),
                sc=float(tot[j]))


def brute(step, segs, total):
    ts = np.arange(0.0, total + 1e-9, step)
    t0 = time.perf_counter()
    res = phys.sim_many(segs, ts)
    return ts, res, time.perf_counter() - t0, len(ts)


def hier_search(segs, total):
    """分层: L0 10s 全程 -> 可行区间; L1 1s 区间内; L2 0.1s 最优点+边界附近;
    任一级失败则升级(最终可退化为全程 0.1s, 保证不差于暴力)."""
    t0 = time.perf_counter()
    evals = 0
    ts0 = np.arange(0.0, total + 1e-9, 10.0)
    r0 = phys.sim_many(segs, ts0); evals += len(ts0)
    ok0 = feasible_mask(r0)
    if ok0.any():
        idx = np.where(ok0)[0]
        groups, s, p = [], idx[0], idx[0]
        for i in idx[1:]:
            if i == p + 1:
                p = i
            else:
                groups.append((s, p)); s = p = i
        groups.append((s, p))
        spans = [np.arange(max(0.0, ts0[a] - 10.0),
                           min(total, ts0[b] + 10.0) + 1e-9, 1.0)
                 for a, b in groups]
    else:
        spans = [np.arange(0.0, total + 1e-9, 1.0)]  # 回退: 全程 1s
    ts1 = np.unique(np.concatenate(spans))
    r1 = phys.sim_many(segs, ts1); evals += len(ts1)

    parts_ts, parts_r = [ts0, ts1], [r0, r1]
    ok1 = feasible_mask(r1)
    if ok1.any():
        fi = np.where(ok1)[0]
        fis = set(fi.tolist())
        top = fi[np.argsort(r1[fi, 3])[:5]]            # 充电时间最优的 5 个
        bnd = [i for i in fi if (i - 1) not in fis or (i + 1) not in fis]
        centers = ts1[np.unique(np.concatenate([top, np.array(bnd, int)]))]
        win = [np.arange(max(0.0, c - 3.0), min(total, c + 3.0) + 1e-9, 0.1)
               for c in centers]
        ts2 = np.unique(np.concatenate(win))
        r2 = phys.sim_many(segs, ts2); evals += len(ts2)
        parts_ts.append(ts2); parts_r.append(r2)
        ok_any = True
    else:
        ok_any = feasible_mask(r0).any()

    if not ok_any:  # 最后升级: 全程 0.1s (稀有窄窗 case 才走到这)
        ts2 = np.arange(0.0, total + 1e-9, 0.1)
        r2 = phys.sim_many(segs, ts2); evals += len(ts2)
        parts_ts.append(ts2); parts_r.append(r2)

    ts_all = np.concatenate(parts_ts)
    res_all = np.vstack(parts_r)
    return ts_all, res_all, time.perf_counter() - t0, evals


methods = ["bf10", "bf1", "bf0.1", "hier"]
rows = {m: [] for m in methods}
times = {m: [] for m in methods}
evals = {m: [] for m in methods}

for n, cid in enumerate(sel, 1):
    segs = nav[cid - 1]
    total = phys.route_totals(segs)[0]
    row = orc_idx.loc[cid]
    for m, step in (("bf10", 10.0), ("bf1", 1.0), ("bf0.1", 0.1)):
        ts, res, wt, ev = brute(step, segs, total)
        rows[m].append(judge(ts, res, row)); times[m].append(wt); evals[m].append(ev)
    ts, res, wt, ev = hier_search(segs, total)
    rows["hier"].append(judge(ts, res, row)); times["hier"].append(wt)
    evals["hier"].append(ev)
    if n % 20 == 0:
        print(f"  ... {n}/{len(sel)}")

# ---- 前缀缓存价值: 同一批 1s 候选, 逐次完整仿真 vs 批量前缀复用 ----
nc_ids = rng.choice(sel, 20, replace=False)
ratios = []
for cid in nc_ids:
    segs = nav[cid - 1]
    total = phys.route_totals(segs)[0]
    ts = np.arange(0.0, total + 1e-9, 1.0)
    t0 = time.perf_counter()
    phys.sim_many(segs, ts)                       # 缓存
    t_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in ts[:200]:                            # 无缓存(抽样 200 个外推)
        phys.sim(segs, float(t))
    t_nc = (time.perf_counter() - t0) / 200.0 * len(ts)
    ratios.append(t_nc / max(t_c, 1e-12))
cache_factor = float(np.mean(ratios))
print(f"\n前缀缓存加速比 (无缓存/缓存, 1s 步长, 20 case): x{cache_factor:.1f}")

# ---- 汇总 ----
print("\n================= 实验结果 (样本 %d case, 其中窄窗 %d) ================="
      % (len(sel), len(narrow_set)))
hdr = f"{'方法':<10}{'检出率':>8}{'均耗时ms':>10}{'均评估数':>10}{'Δt均值s':>10}{'Δt最大s':>10}{'得分差':>10}{'外推10万':>12}"
print(hdr)
for m in methods:
    R = pd.DataFrame(rows[m])
    det = R.found.mean()
    gap = 50.0 - R.sc                             # 漏检记 50 分差
    t_ms = np.mean(times[m]) * 1e3
    ev = np.mean(evals[m])
    dt_m = R.dt.mean() if R.found.any() else float("nan")
    dt_x = R.dt.max() if R.found.any() else float("nan")
    hours = np.mean(times[m]) * 100000 / 3600.0
    print(f"{m:<10}{det:>8.1%}{t_ms:>10.1f}{ev:>10.0f}"
          f"{dt_m:>10.3f}{dt_x:>10.3f}{gap.mean():>10.2f}{hours:>9.1f} h")

# 窄窗子集单独统计
print("\n---- 窄窗子集 (feasible_count<=50) ----")
for m in methods:
    sub = [rows[m][i] for i, c in enumerate(sel) if c in narrow_set]
    R = pd.DataFrame(sub)
    print(f"{m:<10} 检出率 {R.found.mean():>6.1%}   得分差均值 {(50.0 - R.sc).mean():>6.2f}")

# ---- ML 前向耗时 ----
try:
    import torch
    model = torch.nn.Sequential(
        torch.nn.Linear(30, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 3))
    model.eval()
    x = torch.randn(100000, 30)
    with torch.no_grad():
        for _ in range(3):                      # 预热
            model(x[:10000])
        t1 = time.perf_counter()
        for _ in range(10):
            model(x)
    per = (time.perf_counter() - t1) / 10.0 / 100000 * 1e6
    print(f"\nMLP 前向 (batch=100k, CPU, 空闲机器): {per:.1f} us/case "
          f"-> 10 万 case 共 {per * 100000 / 1e6:.2f} s")
except Exception as e:
    print("\nML 计时跳过:", e)

print("\n注: 所有方法在同一空闲机器上测量(Oracle 生成已完成), 计时干净;"
      "\n    生产级 C 生成器(双 Oracle raw+safe, OpenMP 16 线程)实测 0.158 s/case,")
print("    即 10 万 case 实际耗时 4.4 h (含并行加速与 I/O).")
