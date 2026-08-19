# -*- coding: utf-8 -*-
"""
preheat_env.py — 共享物理环境与数据工具

- physics.dll (ctypes): 与 student_solution.c / oracle 生成器完全同源的物理仿真
- nav_train_100000.txt (GB18030) 解析, 缓存为 .npy
- Oracle CSV 加载, 80k/10k/10k 划分 (按 case_id)
- 在线安全层: RL/MLP 预测 -> 物理校验 -> ±30 s 局部 1 s 搜索 -> 必要时全路线回退
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Windows: physics.dll ; Linux/macOS: libphysics.so
_LIB_NAME = "physics.dll" if sys.platform == "win32" else "libphysics.so"
DLL_PATH = ROOT / "oracle" / _LIB_NAME
NAV_PATH = ROOT / "sdk" / "data" / "nav_train_100000.txt"
ORACLE_DIR = ROOT / "oracle" / "output"
CACHE_PATH = ROOT / "ml" / "nav_cache.npy"

# 固定初始状态 (与官方 Mock 评测条件一致)
T_INIT = 0.0        # degC
SOC_INIT = 65.0     # %
DT = 0.5            # s

# 约束 (T_low, T_high, SOC_min)
RAW_BOUNDS = (20.00, 25.00, 10.00)
SAFE_BOUNDS = (20.05, 24.95, 10.05)

TRAIN_END = 80000   # case_id 1..80000
VAL_END = 90000     # case_id 80001..90000 ; test 90001..100000


# ----------------------------------------------------------------------
# physics.dll 包装
# ----------------------------------------------------------------------
class Physics:
    def __init__(self, dll_path: Path = DLL_PATH):
        self.lib = ctypes.CDLL(str(dll_path))
        d, pd_ = ctypes.c_double, ctypes.POINTER(ctypes.c_double)
        self.lib.ph_dll_simulate.restype = ctypes.c_int
        self.lib.ph_dll_simulate.argtypes = [pd_, ctypes.c_int, d, d, d, d, pd_]
        self.lib.ph_dll_simulate_many.restype = ctypes.c_int
        self.lib.ph_dll_simulate_many.argtypes = [pd_, ctypes.c_int, d, d, d,
                                                  pd_, ctypes.c_int, pd_]
        self.lib.ph_route_totals.restype = ctypes.c_int
        self.lib.ph_route_totals.argtypes = [pd_, ctypes.c_int, pd_]
        self._d, self._pd = d, pd_

    def sim(self, segs: np.ndarray, t_start: float,
            t0: float = T_INIT, soc0: float = SOC_INIT,
            dt: float = DT):
        """segs: (n,4) float64 [s_km, v_kmh, P_kW, T_env]. 返回 dict 或 None."""
        segs = np.ascontiguousarray(segs, dtype=np.float64)
        n = segs.shape[0]
        arr = segs.ctypes.data_as(self._pd)
        out = (self._d * 6)()
        ok = self.lib.ph_dll_simulate(arr, n, t0, soc0, dt, float(t_start), out)
        if not ok:
            return None
        return dict(T_end=out[0], soc_end=out[1], E_heat=out[2],
                    charge_time=out[3], charge_power=out[4], dist=out[5])

    def sim_many(self, segs: np.ndarray, t_starts,
                 t0: float = T_INIT, soc0: float = SOC_INIT,
                 dt: float = DT) -> np.ndarray:
        """返回 (len(t_starts), 6); 物理失败的行全为 0."""
        segs = np.ascontiguousarray(segs, dtype=np.float64)
        ts = np.ascontiguousarray(np.asarray(t_starts, dtype=np.float64).ravel())
        n = segs.shape[0]
        out = np.zeros((ts.size, 6), dtype=np.float64)
        self.lib.ph_dll_simulate_many(
            segs.ctypes.data_as(self._pd), n, t0, soc0, dt,
            ts.ctypes.data_as(self._pd), ts.size,
            out.ctypes.data_as(self._pd))
        return out

    def route_totals(self, segs: np.ndarray):
        segs = np.ascontiguousarray(segs, dtype=np.float64)
        out = (self._d * 2)()
        ok = self.lib.ph_route_totals(segs.ctypes.data_as(self._pd),
                                      segs.shape[0], out)
        return (out[0], out[1]) if ok else None


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
_NUM_RE = None  # (保留占位; 解析已改为字节级, 见 load_nav)


def load_nav(path: Path = NAV_PATH, cache: Path = CACHE_PATH) -> np.ndarray:
    """返回 (N, 6, 4) float64: [s_km, v_kmh, P_drive_kW, T_env].

    字节级解析 (case 头 = '=' 开头且含 '|'; 段落行 = ASCII 数字开头, 5 列),
    不依赖 GB18030 解码, 单遍扫描。
    """
    if cache.exists():
        return np.load(cache)
    data = path.read_bytes()
    cases, cur = [], None
    for raw in data.split(b"\n"):
        line = raw.rstrip(b"\r").strip()
        if not line:
            continue
        if line[:1] == b"=" and b"|" in line:
            if cur is not None and len(cur):
                cases.append(cur)
            cur = []
        elif cur is not None and 0x30 <= line[0] <= 0x39:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cur.append([float(x) for x in parts[1:5]])
                except ValueError:
                    pass
    if cur is not None and len(cur):
        cases.append(cur)
    arr = np.zeros((len(cases), 6, 4), dtype=np.float64)
    for i, c in enumerate(cases):
        a = np.asarray(c[:6], dtype=np.float64)
        arr[i, : a.shape[0]] = a
    np.save(cache, arr)
    return arr


def load_oracle(kind: str = "raw",
                prefix: Path = ORACLE_DIR / "oracle") -> pd.DataFrame:
    df = pd.read_csv(prefix.with_name(f"{prefix.name}_{kind}.csv"))
    return df


def split_ids(df: pd.DataFrame):
    train = df[df.case_id <= TRAIN_END]
    val = df[(df.case_id > TRAIN_END) & (df.case_id <= VAL_END)]
    test = df[df.case_id > VAL_END]
    return train, val, test


# ----------------------------------------------------------------------
# 评分与可行性
# ----------------------------------------------------------------------
def is_feasible(T_end: float, soc_end: float,
                bounds=RAW_BOUNDS) -> bool:
    lo, hi, soc_min = bounds
    return lo <= T_end <= hi and soc_end >= soc_min


def score(tch: float, E: float, tch_min: float, tch_max: float,
          e_min: float, e_max: float) -> float:
    """比赛公式: 30 分充电时间 + 20 分能耗 (可行集内 min-max 归一化).

    上界封顶 30/20: 真实评测中策略必在可行集内, 数学上不会超过满分;
    离线复算时全精度仿真值可能略低于 CSV 舍入后的 min, 封顶可免疫
    窄窗 case 的精度伪影 (见 case 5173: 曾算出 50.85).
    下界不封: 策略差于可行集内最差候选时保留负分信号.
    """
    cr = tch_max - tch_min
    er = e_max - e_min
    s_t = min(30.0, 30.0 * (tch_max - tch) / cr) if cr > 1e-9 else 30.0
    s_e = min(20.0, 20.0 * (e_max - E) / er) if er > 1e-9 else 20.0
    return s_t + s_e


def _pick_best_index(res: np.ndarray, bounds) -> int:
    """在 (n,6) 结果里选最优可行行: 先可行, 再最小充电时间, 再最小能耗."""
    lo, hi, soc_min = bounds
    T, S, E, C = res[:, 0], res[:, 1], res[:, 2], res[:, 3]
    feas = (T >= lo) & (T <= hi) & (S >= soc_min) & (C > 0)
    if not feas.any():
        return -1
    idx = np.where(feas)[0]
    # 充电时间最小优先 (30 分权重), 并列取能耗小
    order = np.lexsort((E[idx], C[idx]))
    return int(idx[order[0]])


def _penalty(res: np.ndarray, bounds) -> np.ndarray:
    """约束违约惩罚 (与 reward_fn 同权重): tv^2 + 10*sv^2."""
    lo, hi, soc_min = bounds
    T, S = res[:, 0], res[:, 1]
    tv = np.maximum(0.0, np.maximum(lo - T, T - hi))
    sv = np.maximum(0.0, soc_min - S)
    return tv * tv + 10.0 * sv * sv


def safety_layer(phys: Physics, segs: np.ndarray, t_pred: float,
                 total_time: float, bounds=RAW_BOUNDS,
                 window: float = 30.0, fine: float = 1.0):
    """在线安全层.

    返回 (t_final, mode):
      direct   — 预测点直接可行
      local    — ±window 内 1 s 局部搜索找到可行
      fallback — 全路线 10 s 粗搜 + 1 s 细搜找到可行; 窗 <10s 时惩罚引导
                逐级细化 (1s -> 0.1s) 兑住窄可行窗
      none     — 物理上无可行解
    """
    r = phys.sim(segs, t_pred)
    if r is not None and is_feasible(r["T_end"], r["soc_end"], bounds):
        return t_pred, "direct"

    # 局部搜索: ±window, 步长 fine
    lo_t = max(0.0, t_pred - window)
    hi_t = min(total_time, t_pred + window)
    ts = np.arange(lo_t, hi_t + 1e-9, fine)
    res = phys.sim_many(segs, ts)
    k = _pick_best_index(res, bounds)
    if k >= 0:
        return float(ts[k]), "local"

    # 回退: 全路线 10 s 粗搜
    ts = np.arange(0.0, total_time + 1e-9, 10.0)
    res = phys.sim_many(segs, ts)
    k = _pick_best_index(res, bounds)
    if k < 0:
        # 窄可行窗 (<10s 宽, 10s 网格全落窗外): 惩罚引导逐级细化。
        # 最小违约点必贴可行窗边界 (Oracle 生成器同款策略), 1s -> 0.1s 收缩。
        for span, step in ((12.0, 1.0), (2.5, 0.1)):
            c = float(ts[int(np.argmin(_penalty(res, bounds)))])
            ts = np.arange(max(0.0, c - span), min(total_time, c + span) + 1e-9,
                           step)
            res = phys.sim_many(segs, ts)
            k = _pick_best_index(res, bounds)
            if k >= 0:
                return float(ts[k]), "fallback"
        return t_pred, "none"
    # 在粗搜最优点附近 1 s 细搜
    c = float(ts[k])
    lo_t = max(0.0, c - 15.0)
    hi_t = min(total_time, c + 15.0)
    ts2 = np.arange(lo_t, hi_t + 1e-9, fine)
    res2 = phys.sim_many(segs, ts2)
    k2 = _pick_best_index(res2, bounds)
    return (float(ts2[k2]) if k2 >= 0 else c), "fallback"
