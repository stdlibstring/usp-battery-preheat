# -*- coding: utf-8 -*-
"""cmp_robust_mock.py - 扰动测试: (2) 与 (4) 的决策点在物理参数扰动下谁更稳
决策点按 '距充电桩剩余里程' 定义, 对速度/距离扰动天然不变.
"""
import numpy as np
from preheat_env import Physics, RAW_BOUNDS

segs0 = np.array([
    [2.595, 24.830, 13.720, -7.970],
    [6.995, 48.080, 20.180, -8.750],
    [2.762, 24.900, 14.100, -8.060],
    [6.755, 51.790, 19.120, -9.120],
    [17.356, 96.460, 33.070, -10.030],
    [2.537, 24.530, 13.440, -8.020]])

phys = Physics()
lo, hi, smin = RAW_BOUNDS


def dist_to_t(segs, sd):  # sd = 距充电桩剩余里程
    total = segs[:, 0].sum()
    rem = total - sd  # 已驶里程
    t = 0.0
    for s, v, _, _ in segs:
        if rem <= s:
            return t + rem / v * 3600
        rem -= s
        t += s / v * 3600
    return t


def check(segs, sd, label):
    t = dist_to_t(segs, sd)
    r = phys.sim(segs, float(t))
    if r is None:
        print('  %-22s t=%7.1fs  SOC 耗尽/仿真无解                        FAIL(SOC)'
              % (label, t))
        return False
    ok = (lo <= r['T_end'] <= hi) and (r['soc_end'] >= smin)
    print('  %-22s t=%7.1fs  T_end=%6.2f  SOC=%5.2f  E=%.4f  %s'
          % (label, t, r['T_end'], r['soc_end'], r['E_heat'],
             'PASS' if ok else 'FAIL(T)' if not (lo <= r['T_end'] <= hi) else 'FAIL(SOC)'))
    return ok


PTS = [('(2) sd=29.64', 29.64), ('(4) sd=31.38', 31.38)]

scenarios = [
    ('名义工况 (基准)',            lambda s: s.copy()),
    ('车速 -10%',                  lambda s: np.column_stack([s[:, 0], s[:, 1] * 0.9, s[:, 2], s[:, 3]])),
    ('车速 -20%',                  lambda s: np.column_stack([s[:, 0], s[:, 1] * 0.8, s[:, 2], s[:, 3]])),
    ('车速 +10%',                  lambda s: np.column_stack([s[:, 0], s[:, 1] * 1.1, s[:, 2], s[:, 3]])),
    ('驱动功率 +15%',              lambda s: np.column_stack([s[:, 0], s[:, 1], s[:, 2] * 1.15, s[:, 3]])),
    ('环温 -3 C',                  lambda s: np.column_stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3] - 3.0])),
    ('环温 +3 C',                  lambda s: np.column_stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3] + 3.0])),
    ('里程 +10% (路更长)',          lambda s: np.column_stack([s[:, 0] * 1.1, s[:, 1], s[:, 2], s[:, 3]])),
    ('车速-10% & 环温-3 (复合)',    lambda s: np.column_stack([s[:, 0], s[:, 1] * 0.9, s[:, 2], s[:, 3] - 3.0])),
]

print('%-28s %s' % ('场景', '决策点表现'))
for name, fn in scenarios:
    segs = fn(segs0)
    print(name)
    for pl, sd in PTS:
        check(segs, sd, pl)
    print()
