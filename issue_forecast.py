"""
GitHub Issue 累积数量趋势分析 & 预测
====================================
流程：
  1. 从 curve.png 提取绿色曲线像素
  2. 像素坐标 → (日期, Issue数量) 标定
  3. 多模型曲线拟合（对数/幂函数/平方根/三参数对数/Logistic/混合）
  4. 自动选优（最高 R²）
  5. 预测未来 +30/60/90/180/365/730 天
  6. 输出拟合对比图 + 预测图（黑色背景风格）

已知锚点（真实值）:
  2025-10-22  →  9634
  2026-04-14  →  12245
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 字体配置（中文支持）
# ─────────────────────────────────────────────
def _setup_font():
    candidates = [
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei",
        "PingFang SC", "Microsoft YaHei", "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            rcParams["font.family"] = c
            return
    # 找系统中任意 CJK 字体
    for f in fm.fontManager.ttflist:
        if any(kw in f.name.lower() for kw in ("cjk", "chinese", "wqy", "noto")):
            rcParams["font.family"] = f.name
            return
    # 回退：关闭中文，用英文标签
    rcParams["font.family"] = "DejaVu Sans"

_setup_font()
rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────
# 1. 图像曲线提取
# ─────────────────────────────────────────────
CURVE_PNG = os.path.join(os.path.dirname(__file__), "curve.png")

# 已知锚点（真实日期 & Issue数量）
ANCHOR1_DATE  = datetime(2025, 10, 22)
ANCHOR1_COUNT = 9634

ANCHOR2_DATE  = datetime(2026, 4, 14)
ANCHOR2_COUNT = 12245

# 图像 X 轴对应日期范围（像素列 0 → col_max）
X_DATE_LEFT  = datetime(2025, 10, 26)
X_DATE_RIGHT = datetime(2026, 4,  9)


def extract_curve(png_path: str) -> pd.DataFrame:
    """
    从图像中提取绿色曲线，返回 DataFrame(date, issue_count)。

    标定策略：
      - X 轴：列像素线性映射到日期范围
      - Y 轴：用两个锚点 + 曲线端点做线性外推标定
    """
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img)

    # ── 提取纯绿色像素 (G 高、R/B 低) ──
    G, R, B = arr[:, :, 1], arr[:, :, 0], arr[:, :, 2]
    mask = (G > 80) & (R < 50) & (B < 50)
    rows_px, cols_px = np.where(mask)

    if len(rows_px) == 0:
        raise ValueError("未检测到绿色曲线像素，请检查阈值或图像路径。")

    # ── 每列取绿色像素行的中位数 ──
    col_min, col_max = int(cols_px.min()), int(cols_px.max())
    curve_cols, curve_rows = [], []
    for c in range(col_min, col_max + 1):
        r_in_col = rows_px[cols_px == c]
        if len(r_in_col) > 0:
            curve_cols.append(c)
            curve_rows.append(int(np.median(r_in_col)))

    curve_cols = np.array(curve_cols, dtype=float)
    curve_rows = np.array(curve_rows, dtype=float)

    # ── X 轴标定：列像素 → 日期 ──
    days_total = (X_DATE_RIGHT - X_DATE_LEFT).days      # 165 天
    col_range  = col_max - col_min                        # 像素列宽
    days_per_px = days_total / col_range
    dates = [X_DATE_LEFT + timedelta(days=float(c - col_min) * days_per_px)
             for c in curve_cols]

    # ── Y 轴标定：利用两个锚点线性外推 ──
    # 将锚点日期映射到对应的"虚拟列像素"并外推行像素
    def date_to_col(d):
        return col_min + (d - X_DATE_LEFT).days / days_per_px

    col_a1 = date_to_col(ANCHOR1_DATE)   # ≈ col_min - 13.6  (图像左侧外)
    col_a2 = date_to_col(ANCHOR2_DATE)   # ≈ col_max + 16.9  (图像右侧外)

    # 用曲线端点作简单线性外推得到锚点处的"预测行"
    slope_px = (curve_rows[-1] - curve_rows[0]) / (curve_cols[-1] - curve_cols[0])

    def extrapolate_row(col):
        return curve_rows[0] + slope_px * (col - curve_cols[0])

    row_a1 = extrapolate_row(col_a1)
    row_a2 = extrapolate_row(col_a2)

    # 解线性方程: count = alpha + beta * row_px
    # alpha + beta * row_a1 = ANCHOR1_COUNT
    # alpha + beta * row_a2 = ANCHOR2_COUNT
    beta  = (ANCHOR1_COUNT - ANCHOR2_COUNT) / (row_a1 - row_a2)
    alpha = ANCHOR1_COUNT - beta * row_a1

    counts = alpha + beta * curve_rows

    df = pd.DataFrame({"date": dates, "issue_count": counts})
    df = df.sort_values("date").reset_index(drop=True)

    # 加入两个锚点（如果不在范围内则追加）
    for d, c in [(ANCHOR1_DATE, ANCHOR1_COUNT), (ANCHOR2_DATE, ANCHOR2_COUNT)]:
        if not ((df["date"] - d).abs() < timedelta(days=1)).any():
            df = pd.concat(
                [df, pd.DataFrame({"date": [d], "issue_count": [float(c)]})],
                ignore_index=True,
            )
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[提取] 共 {len(df)} 个数据点")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  数量范围: {df['issue_count'].min():.0f} ~ {df['issue_count'].max():.0f}")
    return df


# ─────────────────────────────────────────────
# 2. 模型定义
# ─────────────────────────────────────────────
def _t(dates, origin):
    """将日期序列转换为从 origin 起算的天数（浮点）。"""
    return np.array([(d - origin).days for d in dates], dtype=float)


ORIGIN = ANCHOR1_DATE   # t=0 对应 2025-10-22


def model_log2(t, a, b):
    """两参数对数: a + b * ln(t+1)"""
    return a + b * np.log(t + 1)

def model_power(t, a, b, c):
    """幂函数: a + b * t^c"""
    return a + b * np.power(np.maximum(t, 0), c)

def model_sqrt(t, a, b):
    """平方根: a + b * sqrt(t)"""
    return a + b * np.sqrt(np.maximum(t, 0))

def model_log3(t, a, b, c):
    """三参数对数: a + b * ln(t + c)"""
    return a + b * np.log(np.maximum(t + c, 1e-6))

def model_logistic(t, L, k, t0):
    """Logistic（S 形）: L / (1 + exp(-k*(t-t0)))"""
    return L / (1.0 + np.exp(-k * (t - t0)))

def model_mixed(t, a, b, c):
    """混合: a + b * sqrt(t) + c * ln(t+1)"""
    return a + b * np.sqrt(np.maximum(t, 0)) + c * np.log(t + 1)

def model_exp_decay(t, a, b, c):
    """减速指数: a - b * exp(-c * t)"""
    return a - b * np.exp(-c * np.maximum(t, 0))


MODELS = {
    "两参数对数": (model_log2,   [9000,  500],           {}),
    "幂函数":     (model_power,  [9000, 100, 0.3],        {}),
    "平方根":     (model_sqrt,   [9000, 200],             {}),
    "三参数对数": (model_log3,   [5000, 1500, 5],         {}),
    "Logistic":   (model_logistic,[13000, 0.01, 200],     {}),
    "混合模型":   (model_mixed,  [9000, 100, 200],        {}),
    "减速指数":   (model_exp_decay,[13000, 4000, 0.02],   {}),
}


# ─────────────────────────────────────────────
# 3. 拟合 & 选优
# ─────────────────────────────────────────────
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_all_models(df: pd.DataFrame):
    t_vals   = _t(df["date"], ORIGIN)
    y_vals   = df["issue_count"].values
    results  = {}

    for name, (func, p0, kwargs) in MODELS.items():
        try:
            popt, _ = curve_fit(
                func, t_vals, y_vals, p0=p0,
                maxfev=10000,
                bounds=(-np.inf, np.inf),
                **kwargs,
            )
            y_pred = func(t_vals, *popt)
            r2     = r2_score(y_vals, y_pred)
            results[name] = {"func": func, "popt": popt, "r2": r2}
            print(f"  {name:12s}: R²={r2:.4f}  params={np.round(popt, 4)}")
        except Exception as e:
            print(f"  {name:12s}: 拟合失败 ({e})")

    best_name = max(results, key=lambda n: results[n]["r2"])
    print(f"\n[最优模型] {best_name}  R²={results[best_name]['r2']:.4f}")
    return results, best_name


# ─────────────────────────────────────────────
# 4. 预测
# ─────────────────────────────────────────────
FUTURE_DAYS = [30, 60, 90, 180, 365, 730]


def make_predictions(results, best_name, ref_date=None):
    if ref_date is None:
        ref_date = ANCHOR2_DATE   # 以当前日期 2026-04-14 为起点预测

    preds = {}
    for name, info in results.items():
        preds[name] = {}
        for fd in FUTURE_DAYS:
            future_date = ref_date + timedelta(days=fd)
            t_fut       = (future_date - ORIGIN).days
            val         = info["func"](t_fut, *info["popt"])
            preds[name][fd] = (future_date, int(round(val)))

    # 打印预测表
    print("\n" + "=" * 72)
    print(f"{'预测日期':^12}{'距今天数':^10}", end="")
    for n in results:
        print(f"{n:^14}", end="")
    print()
    print("-" * 72)
    for fd in FUTURE_DAYS:
        fd_date = ref_date + timedelta(days=fd)
        print(f"{str(fd_date.date()):^12}{fd:^10}", end="")
        for n in results:
            print(f"{preds[n][fd][1]:^14,}", end="")
        print()
    print("=" * 72)

    # 最优模型单独展示
    print(f"\n[最优模型 {best_name}] 预测结果:")
    print(f"{'参考日期':12s} {ref_date.date()}  (Issue数量 ≈ {ANCHOR2_COUNT})")
    print(f"{'日期':12s} {'+天数':6s} {'预测数量':>10s}")
    print("-" * 35)
    for fd in FUTURE_DAYS:
        fd_date, val = preds[best_name][fd]
        print(f"{str(fd_date.date()):12s} {'+'+str(fd):6s} {val:>10,}")

    return preds


# ─────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────
DARK_BG   = "#0d1117"
GREEN_LINE = "#39d353"
GREEN_LITE = "#7ee787"
CYAN_LINE  = "#58a6ff"
ORANGE_LINE= "#f78166"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#e6edf3"
MUTED_TEXT = "#8b949e"


def _dark_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)


def plot_fit_comparison(df: pd.DataFrame, results: dict, best_name: str, out_path: str):
    """图1：拟合对比图（所有模型 vs 原始数据）"""
    t_vals  = _t(df["date"], ORIGIN)
    y_vals  = df["issue_count"].values

    t_smooth = np.linspace(t_vals.min(), t_vals.max(), 500)
    smooth_dates = [ORIGIN + timedelta(days=float(t)) for t in t_smooth]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    # 原始提取数据
    ax.scatter(df["date"], y_vals, color=GREEN_LITE, s=3, alpha=0.6,
               zorder=3, label="提取数据点")

    # 锚点
    for d, c, lbl in [
        (ANCHOR1_DATE, ANCHOR1_COUNT, f"锚点1 ({ANCHOR1_DATE.date()})"),
        (ANCHOR2_DATE, ANCHOR2_COUNT, f"锚点2 ({ANCHOR2_DATE.date()})"),
    ]:
        ax.scatter([d], [c], color="#ff7b72", s=80, zorder=5)
        ax.annotate(f"{lbl}\n{c:,}", (d, c),
                    textcoords="offset points", xytext=(8, -14),
                    color="#ff7b72", fontsize=8)

    # 各模型拟合曲线
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, info), color in zip(results.items(), colors):
        y_smooth = info["func"](t_smooth, *info["popt"])
        lw = 2.5 if name == best_name else 1.0
        ls = "-" if name == best_name else "--"
        label = f"{'★ ' if name==best_name else ''}{name} (R²={info['r2']:.4f})"
        ax.plot(smooth_dates, y_smooth, color=color, lw=lw, ls=ls,
                label=label, zorder=4 if name == best_name else 2)

    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("Issue 累积数量", fontsize=11)
    ax.set_title("GitHub Issue 趋势 — 多模型拟合对比", fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=8.5,
              facecolor="#161b22", edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.85)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[图表] 拟合对比图 → {out_path}")


def plot_forecast(df: pd.DataFrame, results: dict, best_name: str,
                  preds: dict, out_path: str):
    """图2：最优模型预测图（含未来趋势）"""
    info   = results[best_name]
    t_min  = _t([df["date"].min()], ORIGIN)[0]
    # 延伸到 +730 天
    t_end  = (ANCHOR2_DATE + timedelta(days=730) - ORIGIN).days
    t_all  = np.linspace(t_min, t_end, 800)
    dates_all = [ORIGIN + timedelta(days=float(t)) for t in t_all]
    y_all  = info["func"](t_all, *info["popt"])

    t_vals = _t(df["date"], ORIGIN)
    y_vals = df["issue_count"].values

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    # 历史数据区域
    hist_cutoff = ANCHOR2_DATE
    hist_mask   = np.array(dates_all) <= hist_cutoff
    pred_mask   = np.array(dates_all) >= hist_cutoff

    ax.plot(np.array(dates_all)[hist_mask], y_all[hist_mask],
            color=GREEN_LINE, lw=2, label=f"历史拟合 ({best_name})")
    ax.plot(np.array(dates_all)[pred_mask], y_all[pred_mask],
            color=CYAN_LINE, lw=2, ls="--", label="预测趋势")

    # 原始提取散点
    ax.scatter(df["date"], y_vals, color=GREEN_LITE, s=3, alpha=0.4, zorder=3)

    # 锚点
    for d, c in [(ANCHOR1_DATE, ANCHOR1_COUNT), (ANCHOR2_DATE, ANCHOR2_COUNT)]:
        ax.scatter([d], [c], color=ORANGE_LINE, s=80, zorder=5)

    # 预测点标注
    for fd in FUTURE_DAYS:
        fd_date, val = preds[best_name][fd]
        ax.scatter([fd_date], [val], color=CYAN_LINE, s=45, zorder=6)
        ax.annotate(
            f"+{fd}d\n{val:,}",
            (fd_date, val),
            textcoords="offset points",
            xytext=(5, 6),
            color=CYAN_LINE,
            fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=CYAN_LINE, lw=0.6),
        )

    # 当前日期竖线
    ax.axvline(ANCHOR2_DATE, color=ORANGE_LINE, lw=1, ls=":", alpha=0.8,
               label=f"今日 ({ANCHOR2_DATE.date()})")

    ax.set_xlabel("日期", fontsize=11)
    ax.set_ylabel("Issue 累积数量", fontsize=11)
    ax.set_title(
        f"GitHub Issue 趋势预测  [{best_name}  R²={info['r2']:.4f}]",
        fontsize=13, pad=12,
    )
    ax.legend(loc="upper left", fontsize=9,
              facecolor="#161b22", edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.85)

    # 右侧 Y 轴（百分比增幅）
    base = ANCHOR2_COUNT
    ax2 = ax.twinx()
    ax2.set_facecolor(DARK_BG)
    ax2.tick_params(colors=MUTED_TEXT, labelsize=8)
    ylim = ax.get_ylim()
    ax2.set_ylim([(y - base) / base * 100 for y in ylim])
    ax2.set_ylabel("相对增幅 (%)", color=MUTED_TEXT, fontsize=9)
    ax2.spines["right"].set_edgecolor(GRID_COLOR)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[图表] 预测图       → {out_path}")


def plot_model_comparison_bar(results: dict, out_path: str):
    """图3：各模型 R² 对比柱状图"""
    names  = list(results.keys())
    r2vals = [results[n]["r2"] for n in names]
    sorted_idx = np.argsort(r2vals)[::-1]
    names_s  = [names[i] for i in sorted_idx]
    r2vals_s = [r2vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    colors = [GREEN_LINE if r >= max(r2vals) - 0.001 else CYAN_LINE for r in r2vals_s]
    bars = ax.barh(names_s, r2vals_s, color=colors, edgecolor=GRID_COLOR, height=0.55)
    for bar, val in zip(bars, r2vals_s):
        ax.text(val - 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="right",
                color=DARK_BG, fontsize=9, fontweight="bold")
    ax.set_xlim(min(r2vals_s) * 0.98, 1.005)
    ax.set_xlabel("R²", fontsize=10)
    ax.set_title("各模型拟合优度对比 (R²)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[图表] R² 对比图    → {out_path}")


# ─────────────────────────────────────────────
# 6. 保存 CSV
# ─────────────────────────────────────────────
def save_csv(df: pd.DataFrame, results: dict, best_name: str, preds: dict):
    base_dir = os.path.dirname(__file__)

    # 提取数据
    df.to_csv(os.path.join(base_dir, "extracted_data.csv"), index=False,
              date_format="%Y-%m-%d")
    print("[CSV] extracted_data.csv 已保存")

    # 预测表
    rows = []
    for fd in FUTURE_DAYS:
        row = {"days_ahead": fd}
        for n in results:
            fd_date, val = preds[n][fd]
            if "date" not in row:
                row["date"] = str(fd_date.date())
            row[n] = val
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(base_dir, "predictions.csv"), index=False)
    print("[CSV] predictions.csv 已保存")


# ─────────────────────────────────────────────
# 7. 主流程
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GitHub Issue 趋势分析 & 预测")
    print("=" * 60)

    print("\n[步骤 1] 从图像提取曲线数据 ...")
    df = extract_curve(CURVE_PNG)

    print("\n[步骤 2] 多模型曲线拟合 ...")
    results, best_name = fit_all_models(df)

    print("\n[步骤 3] 生成预测 ...")
    preds = make_predictions(results, best_name)

    print("\n[步骤 4] 保存 CSV ...")
    save_csv(df, results, best_name, preds)

    base_dir = os.path.dirname(__file__)
    print("\n[步骤 5] 生成图表 ...")
    plot_fit_comparison(
        df, results, best_name,
        os.path.join(base_dir, "fit_comparison.png"),
    )
    plot_forecast(
        df, results, best_name, preds,
        os.path.join(base_dir, "forecast.png"),
    )
    plot_model_comparison_bar(
        results,
        os.path.join(base_dir, "model_r2_comparison.png"),
    )

    print("\n完成！输出文件：")
    for f in ["extracted_data.csv", "predictions.csv",
              "fit_comparison.png", "forecast.png", "model_r2_comparison.png"]:
        fp = os.path.join(base_dir, f)
        if os.path.exists(fp):
            print(f"  ✓ {f}")


if __name__ == "__main__":
    main()
