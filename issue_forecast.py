"""
GitHub Issue 累积数量趋势分析 & 预测
====================================
通用 CLI 工具：上传曲线图 + 提供左右边缘的日期/数量 → 自动完成提取、拟合、预测、可视化。

用法示例
--------
python issue_forecast.py \\
    --image curve.png \\
    --left-date  2025-10-26 --left-count  9650 \\
    --right-date 2026-04-09 --right-count 12200

可选参数
--------
--today      YYYY-MM-DD   预测起始日期（默认=右边缘日期）
--out-dir    PATH         输出目录（默认=图像所在目录）
--future-days 30 60 90   自定义预测天数列表（默认：30 60 90 180 365 730）

流程 & 输出文件
--------------
步骤 1  提取曲线  →  extracted_data.csv
步骤 2  拟合模型  →  fit_comparison.png, model_r2_comparison.png
步骤 3  生成预测  →  predictions.csv, forecast.png
"""

import argparse
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from PIL import Image
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Pixel thresholds for green curve detection
# ─────────────────────────────────────────────
GREEN_CHANNEL_MIN    = 80   # G channel must be above this
RED_BLUE_CHANNEL_MAX = 50   # R and B channels must be below this

# Small constant to avoid log(0)
LOG_EPSILON = 1e-6

# ─────────────────────────────────────────────
# Visual style constants
# ─────────────────────────────────────────────
DARK_BG     = "#0d1117"
GREEN_LINE  = "#39d353"
GREEN_LITE  = "#7ee787"
CYAN_LINE   = "#58a6ff"
ORANGE_LINE = "#f78166"
GRID_COLOR  = "#21262d"
TEXT_COLOR  = "#e6edf3"
MUTED_TEXT  = "#8b949e"


# ─────────────────────────────────────────────
# 0. Font setup — CJK-aware with English fallback
# ─────────────────────────────────────────────
def _setup_font() -> bool:
    """
    Try to configure a CJK-capable font.
    Returns True if a CJK font was found, False otherwise.
    When False, all chart text should use English to avoid □ boxes.
    """
    cjk_candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK TC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "SimHei", "SimSun", "PingFang SC", "Microsoft YaHei",
        "Source Han Sans CN", "Source Han Sans SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}

    for c in cjk_candidates:
        if c in available:
            rcParams["font.family"] = c
            return True

    # Broader scan for any CJK font the system may have
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in ("cjk", "chinese", "wqy", "noto", "simsun", "simhei")):
            rcParams["font.family"] = f.name
            return True

    # No CJK font found — fall back to default; use English labels in charts
    rcParams["font.family"] = "DejaVu Sans"
    return False


_CJK_OK: bool = _setup_font()
rcParams["axes.unicode_minus"] = False


def _L(zh: str, en: str) -> str:
    """Return the Chinese string when a CJK font is available, English otherwise."""
    return zh if _CJK_OK else en


# ─────────────────────────────────────────────
# 1. 图像曲线提取
# ─────────────────────────────────────────────
def extract_curve(
    png_path: str,
    left_date: datetime,
    left_count: float,
    right_date: datetime,
    right_count: float,
    out_csv: str,
) -> pd.DataFrame:
    """
    从图像中提取绿色曲线，返回并保存 DataFrame(date, issue_count)。

    标定策略（直接使用图表边缘两点）:
      X 轴: col_min → left_date,  col_max → right_date
      Y 轴: 左端行像素 → left_count, 右端行像素 → right_count
            线性方程 count = alpha + beta * row_px 求解
    """
    print(f"[提取] 读取图像: {png_path}")
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img)

    # ── 提取纯绿色像素 ──
    G, R, B = arr[:, :, 1], arr[:, :, 0], arr[:, :, 2]
    mask = (
        (G > GREEN_CHANNEL_MIN)
        & (R < RED_BLUE_CHANNEL_MAX)
        & (B < RED_BLUE_CHANNEL_MAX)
    )
    rows_px, cols_px = np.where(mask)

    if len(rows_px) == 0:
        raise ValueError(
            "未检测到绿色曲线像素。请确认图像路径正确，"
            "或调整 GREEN_CHANNEL_MIN / RED_BLUE_CHANNEL_MAX 阈值。"
        )

    # ── 每列取绿色像素行的中位数 ──
    col_min, col_max = int(cols_px.min()), int(cols_px.max())
    curve_cols, curve_rows = [], []
    for c in range(col_min, col_max + 1):
        r_in_col = rows_px[cols_px == c]
        if len(r_in_col) > 0:
            curve_cols.append(c)
            curve_rows.append(float(np.median(r_in_col)))

    curve_cols = np.array(curve_cols, dtype=float)
    curve_rows = np.array(curve_rows, dtype=float)

    # ── X 轴标定: 列像素 → 日期 ──
    days_total  = (right_date - left_date).days
    col_range   = col_max - col_min
    days_per_px = days_total / col_range
    dates = [
        left_date + timedelta(days=float(c - col_min) * days_per_px)
        for c in curve_cols
    ]

    # ── Y 轴标定: 直接用左右端点 ──
    # left end pixel  → left_count
    # right end pixel → right_count
    row_left  = curve_rows[0]
    row_right = curve_rows[-1]
    if abs(row_left - row_right) < 1e-6:
        raise ValueError("左右端像素行相同，无法标定 Y 轴。")

    beta  = (left_count - right_count) / (row_left - row_right)
    alpha = left_count - beta * row_left
    counts = alpha + beta * curve_rows

    df = pd.DataFrame({"date": dates, "issue_count": counts})
    df = df.sort_values("date").reset_index(drop=True)

    # ── 保存 Step-1 输出 ──
    df.to_csv(out_csv, index=False, date_format="%Y-%m-%d")

    print(f"[提取] 共 {len(df)} 个数据点")
    print(f"  日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  数量范围: {df['issue_count'].min():.0f} ~ {df['issue_count'].max():.0f}")
    print(f"[步骤1输出] {out_csv}")
    return df


# ─────────────────────────────────────────────
# 2. 模型定义
# ─────────────────────────────────────────────
def _t(dates, origin: datetime) -> np.ndarray:
    """日期序列 → 从 origin 起算的天数数组。"""
    return np.array([(d - origin).days for d in dates], dtype=float)


def model_log2(t, a, b):
    """两参数对数: a + b·ln(t+1)"""
    return a + b * np.log(t + 1)

def model_power(t, a, b, c):
    """幂函数: a + b·t^c"""
    return a + b * np.power(np.maximum(t, 0), c)

def model_sqrt(t, a, b):
    """平方根: a + b·√t"""
    return a + b * np.sqrt(np.maximum(t, 0))

def model_log3(t, a, b, c):
    """三参数对数: a + b·ln(t+c)"""
    return a + b * np.log(np.maximum(t + c, LOG_EPSILON))

def model_logistic(t, L, k, t0):
    """Logistic（S 形）: L/(1+exp(-k·(t-t0)))"""
    return L / (1.0 + np.exp(-k * (t - t0)))

def model_mixed(t, a, b, c):
    """混合: a + b·√t + c·ln(t+1)"""
    return a + b * np.sqrt(np.maximum(t, 0)) + c * np.log(t + 1)

def model_exp_decay(t, a, b, c):
    """减速指数: a - b·exp(-c·t)"""
    return a - b * np.exp(-c * np.maximum(t, 0))


def _make_models(y0: float, y1: float) -> dict:
    """
    Build model definitions with initial guesses derived from the data range.
    y0 = starting value, y1 = ending value
    """
    delta  = abs(y1 - y0)
    t_half = 90  # assume ~90 days to reach half the total increment
    return {
        _L("两参数对数", "Log-2P"):    (model_log2,       [y0, delta],                    {}),
        _L("幂函数",     "Power"):     (model_power,      [y0, delta * 0.5, 0.4],         {}),
        _L("平方根",     "Sqrt"):      (model_sqrt,       [y0, delta * 0.1],              {}),
        _L("三参数对数", "Log-3P"):    (model_log3,       [y0 * 0.6, delta, 5.0],         {}),
        "Logistic":                    (model_logistic,   [y1 * 1.05, 0.015, t_half],     {}),
        _L("混合模型",   "Mixed"):     (model_mixed,      [y0, delta * 0.05, delta * 0.3],{}),
        _L("减速指数",   "Exp-Decay"): (model_exp_decay,  [y1 * 1.05, delta * 1.2, 0.015],{}),
    }


# ─────────────────────────────────────────────
# 3. 拟合 & 选优
# ─────────────────────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_all_models(
    df: pd.DataFrame,
    origin: datetime,
    out_fit_png: str,
    out_r2_png: str,
) -> tuple[dict, str]:
    """
    拟合所有模型，输出拟合对比图和 R² 对比图。
    返回 (results_dict, best_model_name)
    """
    t_vals = _t(df["date"], origin)
    y_vals = df["issue_count"].values
    models = _make_models(float(y_vals[0]), float(y_vals[-1]))
    results: dict = {}

    print("[拟合] 开始多模型拟合 ...")
    for name, (func, p0, kwargs) in models.items():
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
        except Exception as exc:
            print(f"  {name:12s}: 拟合失败 ({exc})")

    if not results:
        raise RuntimeError("所有模型均拟合失败，请检查输入数据。")

    best_name = max(results, key=lambda n: results[n]["r2"])
    print(f"\n[最优模型] {best_name}  R²={results[best_name]['r2']:.4f}")

    # ── 保存 Step-2 输出 ──
    _plot_fit_comparison(df, results, best_name, origin, out_fit_png)
    _plot_model_r2_bar(results, out_r2_png)

    return results, best_name


# ─────────────────────────────────────────────
# 4. 预测
# ─────────────────────────────────────────────
def make_predictions(
    results: dict,
    best_name: str,
    origin: datetime,
    ref_date: datetime,
    ref_count: int,
    future_days: list[int],
    out_csv: str,
    out_forecast_png: str,
    df: pd.DataFrame,
) -> dict:
    """
    生成所有模型的预测表，保存 CSV 和预测图。
    """
    preds: dict = {}
    for name, info in results.items():
        preds[name] = {}
        for fd in future_days:
            future_date = ref_date + timedelta(days=fd)
            t_fut       = (future_date - origin).days
            val         = info["func"](t_fut, *info["popt"])
            preds[name][fd] = (future_date, int(round(val)))

    # 打印预测表
    col_w = 13
    header_row = f"{'预测日期':^12}{'距今天数':^10}"
    for n in results:
        header_row += f"{n:^{col_w}}"
    print("\n" + "=" * len(header_row))
    print(header_row)
    print("-" * len(header_row))
    for fd in future_days:
        fd_date = ref_date + timedelta(days=fd)
        row = f"{str(fd_date.date()):^12}{fd:^10}"
        for n in results:
            row += f"{preds[n][fd][1]:^{col_w},}"
        print(row)
    print("=" * len(header_row))

    print(f"\n[最优模型 {best_name}] 预测结果:")
    print(f"  参考日期: {ref_date.date()}  (Issue数量 ≈ {ref_count:,})")
    print(f"  {'日期':12s} {'天数':6s} {'预测数量':>10s}")
    print("  " + "-" * 32)
    for fd in future_days:
        fd_date, val = preds[best_name][fd]
        print(f"  {str(fd_date.date()):12s} {'+'+str(fd):6s} {val:>10,}")

    # ── 保存 Step-3 输出: predictions.csv ──
    rows = []
    for fd in future_days:
        row: dict = {"days_ahead": fd, "date": str((ref_date + timedelta(days=fd)).date())}
        for n in results:
            row[n] = preds[n][fd][1]
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)
    print(f"\n[步骤3输出] {out_csv}")

    # ── 保存 Step-3 输出: forecast.png ──
    _plot_forecast(df, results, best_name, origin, preds, future_days,
                   ref_date, ref_count, out_forecast_png)

    return preds


# ─────────────────────────────────────────────
# 5. 可视化（内部函数）
# ─────────────────────────────────────────────
def _dark_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)


def _plot_fit_comparison(
    df: pd.DataFrame,
    results: dict,
    best_name: str,
    origin: datetime,
    out_path: str,
):
    """Step-2 图1：所有模型拟合对比图。"""
    t_vals = _t(df["date"], origin)
    y_vals = df["issue_count"].values

    t_smooth     = np.linspace(t_vals.min(), t_vals.max(), 500)
    smooth_dates = [origin + timedelta(days=float(t)) for t in t_smooth]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    ax.scatter(df["date"], y_vals, color=GREEN_LITE, s=3, alpha=0.6,
               zorder=3, label=_L("提取数据点", "Extracted data"))

    # 标注左右端锚点
    for d, c, lbl in [
        (df["date"].iloc[0],  y_vals[0],  _L(f"左端点 ({df['date'].iloc[0].date()})", f"Left edge ({df['date'].iloc[0].date()})")),
        (df["date"].iloc[-1], y_vals[-1], _L(f"右端点 ({df['date'].iloc[-1].date()})", f"Right edge ({df['date'].iloc[-1].date()})")),
    ]:
        ax.scatter([d], [c], color="#ff7b72", s=80, zorder=5)
        ax.annotate(f"{lbl}\n{c:,.0f}", (d, c),
                    textcoords="offset points", xytext=(8, -14),
                    color="#ff7b72", fontsize=8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, info), color in zip(results.items(), colors):
        y_smooth = info["func"](t_smooth, *info["popt"])
        lw    = 2.5 if name == best_name else 1.0
        ls    = "-"  if name == best_name else "--"
        label = f"{'* ' if name == best_name else ''}{name} (R\u00b2={info['r2']:.4f})"
        ax.plot(smooth_dates, y_smooth, color=color, lw=lw, ls=ls,
                label=label, zorder=4 if name == best_name else 2)

    ax.set_xlabel(_L("日期", "Date"), fontsize=11)
    ax.set_ylabel(_L("Issue 累积数量", "Cumulative Issues"), fontsize=11)
    ax.set_title(_L("Issue 趋势 — 多模型拟合对比", "Issue Trend - Multi-model Fit Comparison"), fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=8.5,
              facecolor="#161b22", edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.85)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[步骤2输出] {out_path}")


def _plot_model_r2_bar(results: dict, out_path: str):
    """Step-2 图2：各模型 R² 对比柱状图。"""
    names     = list(results.keys())
    r2vals    = [results[n]["r2"] for n in names]
    idx       = np.argsort(r2vals)[::-1]
    names_s   = [names[i] for i in idx]
    r2vals_s  = [r2vals[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    bar_colors = [GREEN_LINE if r >= max(r2vals) - 0.001 else CYAN_LINE
                  for r in r2vals_s]
    bars = ax.barh(names_s, r2vals_s, color=bar_colors,
                   edgecolor=GRID_COLOR, height=0.55)
    for bar, val in zip(bars, r2vals_s):
        ax.text(val - 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="right",
                color=DARK_BG, fontsize=9, fontweight="bold")
    ax.set_xlim(min(r2vals_s) * 0.98, 1.005)
    ax.set_xlabel(_L("R²", "R\u00b2"), fontsize=10)
    ax.set_title(_L("各模型拟合优度对比 (R²)", "Model Goodness-of-Fit Comparison (R\u00b2)"), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[步骤2输出] {out_path}")


def _plot_forecast(
    df: pd.DataFrame,
    results: dict,
    best_name: str,
    origin: datetime,
    preds: dict,
    future_days: list[int],
    ref_date: datetime,
    ref_count: int,
    out_path: str,
):
    """Step-3：最优模型预测图（历史拟合 + 未来趋势）。"""
    info    = results[best_name]
    t_min   = _t([df["date"].min()], origin)[0]
    t_end   = (ref_date + timedelta(days=max(future_days)) - origin).days
    t_all   = np.linspace(t_min, t_end, 800)
    dates_all = [origin + timedelta(days=float(t)) for t in t_all]
    y_all   = info["func"](t_all, *info["popt"])
    y_all   = np.where(np.isfinite(y_all), y_all, np.nan)  # neutralise Inf/NaN before plotting

    t_vals  = _t(df["date"], origin)
    y_vals  = df["issue_count"].values

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    _dark_axes(ax)

    hist_mask = np.array(dates_all) <= ref_date
    pred_mask = np.array(dates_all) >= ref_date

    ax.plot(np.array(dates_all)[hist_mask], y_all[hist_mask],
            color=GREEN_LINE, lw=2, label=_L(f"历史拟合 ({best_name})", f"Historical fit ({best_name})"))
    ax.plot(np.array(dates_all)[pred_mask], y_all[pred_mask],
            color=CYAN_LINE, lw=2, ls="--", label=_L("预测趋势", "Forecast trend"))

    ax.scatter(df["date"], y_vals, color=GREEN_LITE, s=3, alpha=0.4, zorder=3)

    for fd in future_days:
        fd_date, val = preds[best_name][fd]
        ax.scatter([fd_date], [val], color=CYAN_LINE, s=45, zorder=6)
        ax.annotate(
            f"+{fd}d\n{val:,}",
            (fd_date, val),
            textcoords="offset points", xytext=(5, 6),
            color=CYAN_LINE, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=CYAN_LINE, lw=0.6),
        )

    ax.axvline(ref_date, color=ORANGE_LINE, lw=1, ls=":", alpha=0.8,
               label=_L(f"预测起点 ({ref_date.date()})", f"Forecast start ({ref_date.date()})"))

    ax.set_xlabel(_L("日期", "Date"), fontsize=11)
    ax.set_ylabel(_L("Issue 累积数量", "Cumulative Issues"), fontsize=11)
    ax.set_title(
        _L(
            f"Issue 趋势预测  [{best_name}  R\u00b2={info['r2']:.4f}]",
            f"Issue Trend Forecast  [{best_name}  R\u00b2={info['r2']:.4f}]",
        ),
        fontsize=13, pad=12,
    )
    ax.legend(loc="upper left", fontsize=9,
              facecolor="#161b22", edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.85)

    # 右轴：相对增幅
    ax2 = ax.twinx()
    ax2.set_facecolor(DARK_BG)
    ax2.tick_params(colors=MUTED_TEXT, labelsize=8)
    ylim = ax.get_ylim()
    if ref_count and ref_count != 0 and all(np.isfinite(y) for y in ylim):
        ax2.set_ylim([(y - ref_count) / ref_count * 100 for y in ylim])
    ax2.set_ylabel(_L("相对增幅 (%)", "Relative Growth (%)"), color=MUTED_TEXT, fontsize=9)
    ax2.spines["right"].set_edgecolor(GRID_COLOR)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"[步骤3输出] {out_path}")


# ─────────────────────────────────────────────
# 6. CLI 入口
# ─────────────────────────────────────────────
def _parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="从曲线图提取数据、拟合模型并预测未来 Issue 数量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image", default=os.path.join(script_dir, "curve.png"),
        help="曲线图路径（默认: curve.png，与脚本同目录）",
    )
    parser.add_argument(
        "--left-date", required=True, metavar="YYYY-MM-DD",
        help="图表 X 轴左边缘对应的日期",
    )
    parser.add_argument(
        "--left-count", required=True, type=float, metavar="N",
        help="图表左边缘处的真实 Issue 数量",
    )
    parser.add_argument(
        "--right-date", required=True, metavar="YYYY-MM-DD",
        help="图表 X 轴右边缘对应的日期",
    )
    parser.add_argument(
        "--right-count", required=True, type=float, metavar="N",
        help="图表右边缘处的真实 Issue 数量",
    )
    parser.add_argument(
        "--today", default=None, metavar="YYYY-MM-DD",
        help="预测起始日期（默认=右边缘日期）",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="PATH",
        help="输出目录（默认=图像所在目录）",
    )
    parser.add_argument(
        "--future-days", nargs="+", type=int,
        default=[30, 60, 90, 180, 365, 730],
        metavar="N",
        help="预测天数列表（默认: 30 60 90 180 365 730）",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # ── 解析参数 ──
    image_path  = os.path.abspath(args.image)
    left_date   = datetime.strptime(args.left_date,  "%Y-%m-%d")
    left_count  = args.left_count
    right_date  = datetime.strptime(args.right_date, "%Y-%m-%d")
    right_count = args.right_count
    today       = datetime.strptime(args.today, "%Y-%m-%d") if args.today else right_date
    out_dir     = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(image_path)
    future_days = sorted(args.future_days)
    origin      = left_date   # t=0 基准

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Issue 趋势分析 & 预测")
    print("=" * 60)
    print(f"  图像    : {image_path}")
    print(f"  左边缘  : {left_date.date()}  →  {int(left_count):,}")
    print(f"  右边缘  : {right_date.date()}  →  {int(right_count):,}")
    print(f"  预测起点: {today.date()}")
    print(f"  输出目录: {out_dir}")
    print()

    # ── 步骤 1: 提取曲线 ──
    print("─" * 40)
    print("[步骤 1] 从图像提取曲线数据 ...")
    out_extracted = os.path.join(out_dir, "extracted_data.csv")
    df = extract_curve(image_path, left_date, left_count,
                       right_date, right_count, out_extracted)

    # ── 步骤 2: 拟合 ──
    print()
    print("─" * 40)
    print("[步骤 2] 多模型曲线拟合 ...")
    out_fit_png = os.path.join(out_dir, "fit_comparison.png")
    out_r2_png  = os.path.join(out_dir, "model_r2_comparison.png")
    results, best_name = fit_all_models(df, origin, out_fit_png, out_r2_png)

    # ── 步骤 3: 预测 ──
    print()
    print("─" * 40)
    print("[步骤 3] 生成预测 ...")
    out_pred_csv     = os.path.join(out_dir, "predictions.csv")
    out_forecast_png = os.path.join(out_dir, "forecast.png")
    make_predictions(
        results, best_name, origin,
        today, int(right_count), future_days,
        out_pred_csv, out_forecast_png, df,
    )

    # ── 汇总 ──
    print()
    print("=" * 60)
    print("完成！所有输出文件：")
    output_files = [
        ("步骤1", "extracted_data.csv",      "提取的曲线数据点"),
        ("步骤2", "fit_comparison.png",       "多模型拟合对比图"),
        ("步骤2", "model_r2_comparison.png",  "各模型 R² 对比图"),
        ("步骤3", "predictions.csv",          "各模型未来预测表"),
        ("步骤3", "forecast.png",             "最优模型预测可视化"),
    ]
    for step, fname, desc in output_files:
        fp = os.path.join(out_dir, fname)
        status = "✓" if os.path.exists(fp) else "✗"
        print(f"  [{step}] {status}  {fname:35s}  {desc}")
    print("=" * 60)


if __name__ == "__main__":
    main()
