# -*- coding: utf-8 -*-
"""plot_narrative.py - 叙事文档《从替代物理搜索到加速物理搜索》配图

4 张图, 统一风格: Microsoft YaHei / 去顶右边框 / 浅网格 / 数据标签。
数据来源:
  fig1/fig2: ml/plots/direct_eval.json (eval_direct.py, test 集 2000 case 抽样)
             + eval_window_search.py 离线结论 (5313 case)
  fig3:      eval_window_search.py (左) + C 端 A/B run_fast vs run_base2 (右)
  fig4:      eval_direct.py (MLP 全体 MAE) + 双可行子集 safe 偏移实测
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

# ---------- 统一风格 ----------
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})
BLUE = "#3B6FB6"     # 主色
RED = "#D9534F"      # 强调/失败
GREEN = "#2E9E5B"    # 甜点/成功
GRAY = "#9AA0A6"     # 中性
DARK = "#333333"

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

direct = json.loads((OUT / "direct_eval.json").read_text(encoding="utf-8"))
mlp = next(r for r in direct if r["model"] == "MLP")
ppo = next(r for r in direct if r["model"] == "PPO")

# ======================================================================
# fig1 四方案对比: x=平均仿真次数(log) y=全体得分差 mean (误差线=P95)
# ======================================================================
fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=200)
schemes = [
    # (name, sims, gap_mean, gap_p95, color)
    ("纯 MLP 直出",        1,    mlp["gap_all_mean"], mlp["gap_all_p95"], RED),
    ("MLP+PPO 直出",       1,    ppo["gap_all_mean"], ppo["gap_all_p95"], "#E8A33D"),
    ("MLP 先验+物理精搜",  298,  1.74,                5.02,               GREEN),
    ("纯物理全扫",         2553, 0.0,                 0.0,                BLUE),
]
# 两个直出方案 x 相同, 稍微错开
offsets = {"纯 MLP 直出": 1.15, "MLP+PPO 直出": 0.87}
for name, sims, gap, p95, color in schemes:
    x = offsets.get(name, sims)
    yerr = [[max(0.0, gap - 0)], [max(0.0, p95 - gap)]]
    ax.errorbar(x, gap, yerr=yerr, fmt="o", ms=13, color=color,
                ecolor=color, elinewidth=2.2, capsize=6, capthick=2.2,
                zorder=3)
    label = f"{gap:.1f}" + ("" if p95 <= 0.01 else f"\n(P95 {p95:.1f})")
    ax.annotate(label, (x, gap), textcoords="offset points",
                xytext=(14, -4), fontsize=11.5, color=DARK, va="center")

ax.annotate("碰壁区\n快但不可靠", xy=(1.15, 17.2), xytext=(8, 27),
            fontsize=12, color=RED, ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax.annotate("甜点区\n8.6x 加速, 仅损 1.7 分", xy=(298, 1.74), xytext=(330, 9.5),
            fontsize=12, color=GREEN, ha="left",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
ax.annotate("纯物理基准\n(慢而准)", xy=(2553, 0.0), xytext=(1500, 5.2),
            fontsize=12, color=BLUE, ha="center",
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))

ax.set_xscale("log")
ax.set_xticks([1, 10, 100, 298, 1000, 2553])
ax.set_xticklabels(["1", "10", "100", "298", "1000", "2553"])
ax.set_xlim(0.6, 6000)
ax.set_ylim(-2, 58)
ax.set_xlabel("平均物理仿真次数 / case（对数轴）")
ax.set_ylabel("平均得分差 vs Oracle 最优（/50 分）")
ax.set_title("四种方案的『速度-质量』全景：混合架构占据甜点区", fontsize=14,
             pad=12, color=DARK)
ax.text(0.99, 0.02,
        "直出 = 神经网络直接输出决策，无物理搜索兜底（不可行 case 记 0 分）\n"
        "混合 = MLP 预测 t_pred，在 ±120s 窗口内物理精搜（5313 case 离线评估）",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
        color="#666666")
despine(ax)
fig.tight_layout()
fig.savefig(OUT / "fig1_four_way.png", bbox_inches="tight")
plt.close(fig)
print("fig1 ok")

# ======================================================================
# fig2 MLP vs PPO 三指标 (PPO 碰壁)
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), dpi=200)
metrics = [
    ("开启时刻 MAE (s)", [mlp["t_mae_mean"], ppo["t_mae_mean"]],
     "{:.1f}", (0, 150), False, "越低越好"),
    ("raw 直通率 (%)", [mlp["direct_pass_rate"], ppo["direct_pass_rate"]],
     "{:.1f}", (0, 108), True, "越高越好"),
    ("可行子集得分差 (/50)", [mlp["gap_feas_mean"], ppo["gap_feas_mean"]],
     "{:.2f}", (0, 13), False, "越低越好"),
]
for ax, (title, vals, fmt, ylim, pct_better, note) in zip(axes, metrics):
    bars = ax.bar(["MLP", "MLP+PPO"], vals, width=0.52,
                  color=[BLUE, "#E8A33D"], edgecolor="none", zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=12, color=DARK, fontweight="bold")
    ax.set_title(title, fontsize=13, color=DARK)
    ax.set_ylim(*ylim)
    ax.text(0.5, 0.92, note, transform=ax.transAxes, ha="center",
            fontsize=10, color="#888888")
    despine(ax)

axes[1].annotate("PPO 唯一胜出的指标\n(把决策推向可行域中央)",
                 xy=(1, ppo["direct_pass_rate"]), xytext=(0.5, 60),
                 fontsize=10.5, color="#B4761F", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#B4761F", lw=1.4))
fig.suptitle("PPO 精调：直通率上升，但精度与选点质量全面劣化（2000 case 直出评估）",
             fontsize=14, color=DARK, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "fig2_ml_vs_ppo.png", bbox_inches="tight")
plt.close(fig)
print("fig2 ok")

# ======================================================================
# fig3 MLP 优势区间: 左=笨重全扫引擎(离线 5313 case), 右=自适应剪枝引擎(mock)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), dpi=200)

# 左: 离线口径 (v1 类引擎: 系统性 1s 全扫)
ax = axes[0]
vals = [2553, 298]
bars = ax.bar(["无 MLP\n(全程 1s 全扫)", "MLP 先验\n(±120s 窗口)"], vals,
              width=0.5, color=[GRAY, GREEN], zorder=3)
for b, v in zip(bars, vals):
    ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=13, fontweight="bold", color=DARK)
ax.annotate("仿真量 -88.3%\n(8.6x 加速)", xy=(0.5, 1450), ha="center",
            fontsize=13, color=GREEN, fontweight="bold")
ax.set_ylim(0, 3000)
ax.set_ylabel("平均仿真次数 / case")
ax.set_title("笨重引擎：粗扫占大头\n(离线评估, 5313 case)", fontsize=12.5)
despine(ax)

# 右: C 端 A/B 口径 (v2 引擎: 自适应逐级精修)
ax = axes[1]
vals = [990, 980]
bars = ax.bar(["无 MLP\n(纯自适应精修)", "MLP 先验\n(窗口内自适应精修)"], vals,
              width=0.5, color=[GRAY, BLUE], zorder=3)
for b, v in zip(bars, vals):
    ax.annotate(f"{v}", (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=13, fontweight="bold", color=DARK)
ax.annotate("仿真量 -1.0%：引擎已足够聪明\nMLP 无用武之地", xy=(0.5, 720),
            ha="center", fontsize=12.5, color=BLUE)
ax.set_ylim(0, 1200)
ax.set_title("剪枝引擎：精修级联占大头\n(C 端 A/B, mock 路线)", fontsize=12.5)
despine(ax)

fig.suptitle("MLP 的优势区间：加速『笨重的系统性扫描』，而非『已经聪明的搜索』",
             fontsize=14, color=DARK, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig3_mlp_gain.png", bbox_inches="tight")
plt.close(fig)
print("fig3 ok")

# ======================================================================
# fig4 误差分解瀑布: MLP 直出 MAE = safe 标签偏移 + 纯预测误差
# ======================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=200)
total = mlp["t_mae_mean"]          # 52.2 s, 全体口径
offset = 3.5                        # safe 标签系统偏移 (双可行子集实测)
resid = total - offset

ax.bar([0], [total], width=0.5, color=BLUE, zorder=3,
       label="MLP 直出 MAE（全体 2000 case）")
ax.bar([1], [offset], width=0.5, color="#7FA8D9", zorder=3,
       label="safe 标签系统偏移（不可消除）")
ax.bar([1], [resid], bottom=[offset], width=0.5, color="#2E5E94", zorder=3,
       label="模型预测误差（受数据/容量限制）")
ax.plot([0.25, 0.75], [total, total], ls="--", lw=1.2, color="#AAAAAA")

ax.annotate(f"{total:.1f} s", (0, total), textcoords="offset points",
            xytext=(0, 6), ha="center", fontsize=13, fontweight="bold",
            color=DARK)
ax.annotate(f"{offset:.1f} s", (1, offset / 2), ha="center", fontsize=11.5,
            color="white", fontweight="bold")
ax.annotate(f"{resid:.1f} s", (1, offset + resid / 2), ha="center",
            fontsize=12, color="white", fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["MLP 直出总误差", "误差构成"], fontsize=12.5)
ax.set_ylabel("开启时刻 MAE (s)")
ax.set_ylim(0, 62)
ax.set_title("误差预算：~50 s 的误差压不到零，但足够圈定 ±120 s 搜索窗口",
             fontsize=13.5, color=DARK, pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=10.5)
ax.text(0.98, 0.55,
        "物理仿真能区分相距 0.2 s 的候选\n神经网络只能区分相距 ~50 s 的候选\n"
        "-> 精度交给物理，速度交给 MLP",
        transform=ax.transAxes, ha="right", fontsize=10.5, color="#555555")
despine(ax)
fig.tight_layout()
fig.savefig(OUT / "fig4_error_budget.png", bbox_inches="tight")
plt.close(fig)
print("fig4 ok")

print("all figures ->", OUT)
