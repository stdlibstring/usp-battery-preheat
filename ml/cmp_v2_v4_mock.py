# -*- coding: utf-8 -*-
"""cmp_v2_v4_mock.py - 用官方物理引擎全扫 mock 路线, 裁决 (2) 与 (4) 的决策点得分"""
import numpy as np
from preheat_env import Physics, score, RAW_BOUNDS

segs = np.array([
    [2.595, 24.830, 13.720, -7.970],
    [6.995, 48.080, 20.180, -8.750],
    [2.762, 24.900, 14.100, -8.060],
    [6.755, 51.790, 19.120, -9.120],
    [17.356, 96.460, 33.070, -10.030],
    [2.537, 24.530, 13.440, -8.020]])

phys = Physics()
total_time = (segs[:, 0] / segs[:, 1]).sum() * 3600

ts = np.arange(0, total_time, 1.0)
res = phys.sim_many(segs, ts)
lo, hi, smin = RAW_BOUNDS
feas = (res[:, 0] >= lo) & (res[:, 0] <= hi) & (res[:, 1] >= smin) & (res[:, 3] > 0)
F = res[feas]
FT = ts[feas]
print('可行点数:', feas.sum(), ' 可行时长(s):', FT.max() - FT.min())
tch_min, tch_max = F[:, 3].min(), F[:, 3].max()
e_min, e_max = F[:, 2].min(), F[:, 2].max()
print('可行集: tch [%.2f, %.2f]  E [%.4f, %.4f]' % (tch_min, tch_max, e_min, e_max))


def score_at(t):
    r = phys.sim(segs, float(t))
    return r, score(r['charge_time'], r['E_heat'], tch_min, tch_max, e_min, e_max)


best_s, best_t = -1, None
for t, row in zip(FT, F):
    s = score(row[3], row[2], tch_min, tch_max, e_min, e_max)
    if s > best_s:
        best_s, best_t = s, t
for t in np.arange(max(0.0, best_t - 6), best_t + 6, 0.2):
    r = phys.sim(segs, float(t))
    s = score(r['charge_time'], r['E_heat'], tch_min, tch_max, e_min, e_max)
    if s > best_s:
        best_s, best_t = s, t
print('全扫最优: t=%.1fs score=%.3f' % (best_t, best_s))


def dist_to_t(sd):
    rem = sd
    t = 0.0
    for s, v, _, _ in segs:
        if rem <= s:
            return t + rem / v * 3600
        rem -= s
        t += s / v * 3600
    return t


total_dist = segs[:, 0].sum()
print('路线总长: %.3f km, 总时长: %.1f s' % (total_dist, total_time))

for name, sd in [('(2)', 29.64), ('(4)', 31.38)]:
    t = dist_to_t(total_dist - sd)  # start_distance 是距充电桩的剩余里程, 反算已驶里程
    r, s = score_at(t)
    print('%s: sd=%.2fkm t=%.1fs T_end=%.2f E=%.4f tch=%.2f score=%.3f (gap %.2f)'
          % (name, sd, t, r['T_end'], r['E_heat'], r['charge_time'], s, best_s - s))
