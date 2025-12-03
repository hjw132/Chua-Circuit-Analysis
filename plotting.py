from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .config import ChuaParams


# ========== 全局绘图样式设置 ==========
# 配置 matplotlib 的全局样式，包括字体、颜色、网格等设置

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Songti SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.format"] = "pdf"


def _ensure_dir(path: str | Path) -> None:
    """如果目录不存在则创建它。
    
    参数:
        path: 目录路径（字符串或 Path 对象）
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _savefig(fig: plt.Figure, outdir: str | Path, filename: str) -> None:
    """辅助函数：同时保存 PDF（矢量图）和 PNG（快速预览）。
    
    参数:
        fig: matplotlib 图形对象
        outdir: 输出目录
        filename: 文件名（不含扩展名，会自动添加 .pdf 和 .png）
    """
    _ensure_dir(outdir)
    outdir = Path(outdir)
    pdf_path = outdir / filename
    fig.savefig(pdf_path, transparent=False)
    # 可选：生成小 PNG 用于 IDE 中的快速预览
    fig.savefig(outdir / (pdf_path.stem + ".png"), dpi=300)


def compute_psd_series(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """使用简单的加窗 FFT 计算类 Welch 功率谱密度（无需外部依赖）。
    
    使用 Hanning 窗对信号进行加窗处理，然后计算 FFT 得到功率谱密度。
    这是一种简化的 Welch 方法，适用于单次时间序列。
    
    参数:
        t: 时间数组
        x: 信号数组（与 t 长度相同）
        
    返回:
        (freq, psd): 元组
        - freq: 频率数组（仅正频率部分）
        - psd: 功率谱密度数组
    """
    dt = float(np.mean(np.diff(t)))  # 平均采样间隔
    n = len(x)
    win = np.hanning(n)  # Hanning 窗
    xw = (x - x.mean()) * win  # 去均值并加窗
    X = np.fft.rfft(xw)  # 实值 FFT（仅正频率）
    freq = np.fft.rfftfreq(n, dt)  # 对应的频率数组
    # 计算功率谱密度（归一化）
    psd = (np.abs(X) ** 2) / (np.sum(win**2)) / (1.0 / dt)
    return freq, psd


# ========== 基本动力学图表 ==========

def plot_time_phase(t: np.ndarray, s: np.ndarray, outdir: str | Path) -> None:
    """绘制状态变量时域波形与相平面投影（2x1 子图布局）。
    
    生成包含两个子图的图表：
    - 上图：状态变量 x(t) 的时间序列
    - 下图：相平面投影 (x, y)
    
    参数:
        t: 时间数组
        s: 状态数组，形状为 (N, 3)，每行是 [x, y, z]
        outdir: 输出目录
        
    输出文件:
        fig_time_phase.pdf 和 fig_time_phase.png
    """
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8))

    axes[0].plot(t, s[:, 0], lw=1.2, color="#1f77b4")
    axes[0].set_title("状态变量时域波形")
    axes[0].set_ylabel("电压 $v_1$")
    axes[0].set_xlabel("时间 $t$")

    axes[1].plot(s[:, 0], s[:, 1], lw=1.2, color="#e76f51", alpha=0.95)
    axes[1].set_title("相平面投影")
    axes[1].set_xlabel("电压 $v_1$")
    axes[1].set_ylabel("电压 $v_2$")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _savefig(fig, outdir, "fig_time_phase.pdf")
    plt.close(fig)


def plot_psd(t: np.ndarray, x: np.ndarray, outdir: str) -> None:
    """绘制功率谱密度（单图）。
    
    计算并绘制信号 x(t) 的功率谱密度，使用半对数坐标（y 轴对数）。
    
    参数:
        t: 时间数组
        x: 信号数组
        outdir: 输出目录
        
    输出文件:
        psd_x.pdf 和 psd_x.png
    """
    freq, psd = compute_psd_series(t, x)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.2))
    ax.semilogy(freq, psd + 1e-22, lw=1.1, color="#1f77b4")
    ax.set_xlabel("频率 $f$")
    ax.set_ylabel("功率谱密度")
    ax.set_title("功率谱密度 (PSD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _savefig(fig, outdir, "psd_x.pdf")
    plt.close(fig)


def plot_residual(t: np.ndarray, r: np.ndarray, outdir: str) -> None:
    """绘制基于 KCL 的残差（半对数坐标）。
    
    绘制物理一致性残差随时间的变化，使用半对数坐标（y 轴对数）以便
    更好地观察残差的数量级变化。
    
    参数:
        t: 时间数组
        r: 残差数组（无穷范数）
        outdir: 输出目录
        
    输出文件:
        residual.pdf 和 residual.png
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.2))
    ax.semilogy(t, r + 1e-30, lw=1.1, color="#2a9d8f")
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("残差上界 $\\|r\\|_\\infty$")
    ax.set_title("KCL 残差分析")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _savefig(fig, outdir, "residual.pdf")
    plt.close(fig)


def plot_attractor_3d(s: np.ndarray, outdir: str | Path) -> None:
    """绘制三维混沌吸引子，沿电感电流（z）着色。
    
    生成蔡氏电路的三维相空间吸引子图，使用颜色编码表示 z 坐标（电感电流）的值。
    为了性能考虑，如果点数过多会进行降采样。
    
    参数:
        s: 状态数组，形状为 (N, 3)，每行是 [x, y, z]
        outdir: 输出目录
        
    输出文件:
        fig3_chua_attractor.pdf 和 fig3_chua_attractor.png
    """
    pts = np.asarray(s)
    if pts.shape[1] < 3:
        raise ValueError("状态数组必须是形状 (N,3) 才能绘制 3D 吸引子。")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # 降采样以提高绘制性能（最多保留 6000 个点）
    stride = max(1, len(z) // 6000)
    x, y, z = x[::stride], y[::stride], z[::stride]

    segments = np.stack(
        [np.column_stack([x[:-1], y[:-1], z[:-1]]), np.column_stack([x[1:], y[1:], z[1:]])],
        axis=1,
    )
    colors = 0.5 * (z[:-1] + z[1:])
    norm = Normalize(vmin=np.min(colors), vmax=np.max(colors))
    lc = Line3DCollection(
        segments,
        cmap=cm.get_cmap("plasma"),
        norm=norm,
        linewidth=0.8,
        alpha=0.95,
    )
    lc.set_array(colors)

    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(lc)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))
    ax.set_xlabel("电压 $v_1$")
    ax.set_ylabel("电压 $v_2$")
    ax.set_zlabel("电感电流 $i_L$")
    ax.set_title("蔡氏电路三维混沌吸引子", pad=14)
    ax.view_init(elev=22, azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("white")
        axis.pane.set_facecolor("white")

    cbar = fig.colorbar(lc, ax=ax, pad=0.08, shrink=0.8)
    cbar.set_label("电感电流 $i_L$", rotation=90)
    fig.tight_layout()
    _savefig(fig, outdir, "fig_attractor.pdf")
    plt.close(fig)


def plot_stats_panels(
    freq: np.ndarray, psd: np.ndarray, lags: np.ndarray, acf: np.ndarray, outdir: str | Path
) -> None:
    """绘制功率谱密度（PSD）与自相关函数（ACF）并排展示。
    
    生成包含两个子图的统计面板：
    - 左图：功率谱密度（半对数坐标）
    - 右图：自相关函数
    
    参数:
        freq: 频率数组
        psd: 功率谱密度数组
        lags: 滞后数组
        acf: 自相关函数数组
        outdir: 输出目录
        
    输出文件:
        fig_stats.pdf 和 fig_stats.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))

    axes[0].semilogy(freq, psd + 1e-22, color="#1f77b4", lw=1.1)
    axes[0].set_title("功率谱密度")
    axes[0].set_xlabel("频率 $f$")
    axes[0].set_ylabel("功率谱密度")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(lags, acf, color="#e76f51", lw=1.1)
    axes[1].axhline(0.0, color="0.6", ls="--", lw=0.8, alpha=0.7)
    axes[1].set_title("自相关函数")
    axes[1].set_xlabel("滞后 $k$")
    axes[1].set_ylabel("ACF")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    _savefig(fig, outdir, "fig_stats.pdf")
    plt.close(fig)


def plot_validation_panels(
    t: np.ndarray,
    x_rk: np.ndarray,
    x_ra: np.ndarray,
    ev_rk: np.ndarray,
    ev_ra: np.ndarray,
    t_res: np.ndarray,
    residual: np.ndarray,
    outdir: str | Path,
) -> None:
    """绘制积分器对比与残差诊断合并图。
    
    生成包含两个子图的验证面板：
    - 上图：RK45 与 Radau 轨迹对比，包含嵌入的差值放大图
    - 下图：KCL 残差分析（半对数坐标）
    
    参数:
        t: 时间数组
        x_rk: RK45 的 x 分量时间序列
        x_ra: Radau 的 x 分量时间序列
        ev_rk: RK45 的 Poincaré 事件时间数组
        ev_ra: Radau 的 Poincaré 事件时间数组
        t_res: 残差对应的时间数组
        residual: 残差数组
        outdir: 输出目录
        
    输出文件:
        fig_validation.pdf 和 fig_validation.png
    """
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=False)

    axes[0].plot(t, x_rk, lw=1.0, color="#1f77b4", label="RK45")
    axes[0].plot(t, x_ra, lw=1.0, ls="--", color="#e76f51", label="Radau")
    for te in ev_rk:
        axes[0].axvline(te, color="#1f77b4", alpha=0.12, lw=0.8)
    for te in ev_ra:
        axes[0].axvline(te, color="#e76f51", alpha=0.12, lw=0.8)
    axes[0].set_ylabel("电压 $v_1$")
    axes[0].set_title("积分器对比（RK45 与 Radau）")
    axes[0].legend(loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    ax_ins = inset_axes(axes[0], width="45%", height="38%", loc="lower left", borderpad=1.2)
    ax_ins.plot(t, x_rk - x_ra, color="#2a9d8f", lw=0.9)
    ax_ins.set_title("差值放大", fontsize=8)
    ax_ins.tick_params(labelsize=7)

    axes[1].semilogy(t_res, residual + 1e-30, color="#2a9d8f", lw=1.1)
    axes[1].set_xlabel("时间 $t$")
    axes[1].set_ylabel("残差上界 $\\|r\\|_\\infty$")
    axes[1].set_title("残差分析（半对数坐标）")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    _savefig(fig, outdir, "fig_validation.pdf")
    plt.close(fig)


# ========== 电路图/方块图包装函数 ==========

def _call_generate_figures(func_name: str, outdir: str) -> None:
    """调用 generate_figures.py 中的函数，并重定向输出目录。
    
    这是一个内部辅助函数，用于调用外部生成图表的函数。
    
    参数:
        func_name: 要调用的函数名
        outdir: 输出目录
    """
    _ensure_dir(outdir)
    try:
        import generate_figures as gf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "generate_figures.py not found; cannot generate matplotlib circuit diagrams"
        ) from exc

    old_dir = getattr(gf, "output_dir", None)
    try:
        gf.output_dir = outdir  # type: ignore[attr-defined]
        func = getattr(gf, func_name, None)
        if func is None:
            raise RuntimeError(f"Function {func_name} not defined in generate_figures")
        func()  # type: ignore[call-arg]
    finally:
        if old_dir is not None:
            gf.output_dir = old_dir  # type: ignore[attr-defined]


def plot_circuit_schematic(outdir: str) -> None:
    """包装函数：使用 matplotlib 生成蔡氏电路原理图。
    
    调用 generate_figures.py 中的函数生成电路原理图。
    
    参数:
        outdir: 输出目录
        
    输出文件:
        fig1_circuit_schematic.png 和 fig1_circuit_schematic.pdf
    """
    _call_generate_figures("draw_circuit_schematic", outdir)


def plot_state_space_block(outdir: str) -> None:
    """包装函数：使用 matplotlib 生成状态空间方块图。
    
    调用 generate_figures.py 中的函数生成状态空间方块图。
    
    参数:
        outdir: 输出目录
        
    输出文件:
        fig2_state_space_block.png 和 fig2_state_space_block.pdf
    """
    _call_generate_figures("draw_state_space_block", outdir)


def plot_circuit_schematic_sd(outdir: str) -> None:
    """使用 schemdraw 生成简单的电路原理图。
    
    使用 schemdraw 库绘制蔡氏电路原理图，包括电容、电阻、电感等元件。
    
    参数:
        outdir: 输出目录
        
    输出文件:
        fig1_circuit_schematic_sd.png 和 fig1_circuit_schematic_sd.pdf
        
    注意:
        需要安装 schemdraw 库。如果未安装，会抛出异常。
    """
    _ensure_dir(outdir)
    try:
        import schemdraw
        import schemdraw.elements as elm
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "schemdraw not installed; cannot generate schemdraw circuit schematic"
        ) from exc

    with schemdraw.Drawing(  # type: ignore[name-defined]
        file=os.path.join(outdir, "fig1_circuit_schematic_sd")
    ) as d:
        v1 = d.add(elm.Dot(open=True))
        d.push()
        d.add(elm.Capacitor(down=True).label("C1", loc="right"))
        d.add(elm.Ground())
        d.pop()

        d.add(elm.Resistor(right=True).label("R"))
        v2 = d.add(elm.Dot(open=True))

        d.push()
        d.add(elm.Capacitor(down=True).label("C2", loc="right"))
        d.add(elm.Ground())
        d.pop()

        d.move_from(v1.end, 0, -1.0)
        d.add(elm.Box(w=2.0, h=1.0).label("PWL g(v1)"))
        d.add(elm.Ground())

        d.move_from(v2.end, 0, -1.0)
        d.add(elm.Inductor(down=True).label("L", loc="right"))
        d.add(elm.Resistor(down=True).label("r", loc="right"))
        d.add(elm.Ground())


def plot_state_space_block_sd(outdir: str, p: Optional[ChuaParams] = None) -> None:
    """使用 schemdraw 生成简单的状态空间/方块图。
    
    使用 schemdraw 库绘制状态空间方块图，展示 x、y、z 三个状态变量的
    积分器和耦合关系。
    
    参数:
        outdir: 输出目录
        p: 参数对象（可选，当前未使用但保留接口一致性）
        
    输出文件:
        fig2_state_space_block_sd.png 和 fig2_state_space_block_sd.pdf
        
    注意:
        需要安装 schemdraw 库。如果未安装，会抛出异常。
    """
    _ensure_dir(outdir)
    try:
        import schemdraw
        import schemdraw.elements as elm
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "schemdraw not installed; cannot generate schemdraw block diagram"
        ) from exc

    with schemdraw.Drawing(  # type: ignore[name-defined]
        file=os.path.join(outdir, "fig2_state_space_block_sd")
    ) as d:
        # x-branch
        sx = d.add(elm.SummingJunction().label("+", loc="center"))
        ix = d.add(elm.Integrator(right=True).label("dx/dt", loc="center"))
        d.add(elm.Line().right().label("x", loc="right"))

        # y-branch
        d.move_from(sx.center, 0, -2.0)
        sy = d.add(elm.SummingJunction().label("+", loc="center"))
        iy = d.add(elm.Integrator(right=True).label("dy/dt", loc="center"))
        d.add(elm.Line().right().label("y", loc="right"))

        # z-branch
        d.move_from(sy.center, 0, -2.0)
        sz = d.add(elm.SummingJunction().label("+", loc="center"))
        iz = d.add(elm.Integrator(right=True).label("dz/dt", loc="center"))
        d.add(elm.Line().right().label("z", loc="right"))


# ========== 动力学诊断与验证图表 ==========

def plot_bifurcation(
    params: np.ndarray,
    x_on_sec: np.ndarray,
    outdir: str,
    xlabel: str = "参数",
) -> None:
    """绘制一维分岔图。
    
    绘制参数扫描的分岔图，横轴为参数值，纵轴为 Poincaré 截面上的 x 坐标。
    每个点代表一次 Poincaré 事件，点的分布模式反映了系统的动力学结构
    （周期、准周期、混沌等）。
    
    参数:
        params: 参数值数组（每个事件对应一个参数值）
        x_on_sec: Poincaré 截面上的 x 坐标数组
        outdir: 输出目录
        xlabel: 横轴标签（默认 "参数"）
        
    输出文件:
        bifurcation.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(params, x_on_sec, "k.", markersize=1.0, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Poincaré 截面上 $x$ 坐标")
    ax.set_title("一维参数扫描分岔图")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "bifurcation.png"), dpi=300)
    plt.close(fig)


def plot_mle(times: np.ndarray, running: np.ndarray, outdir: str) -> None:
    """绘制最大 Lyapunov 指数收敛曲线。
    
    绘制 MLE 的运行估计值随时间的收敛过程。初始阶段通常有较大波动，
    随着积分时间增加逐渐收敛到稳定值。
    
    参数:
        times: 时间数组
        running: 运行估计数组（每个时间点对应的 MLE 估计值）
        outdir: 输出目录
        
    输出文件:
        mle_convergence.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(times, running, lw=1.0)
    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("运行估计 $\\hat{\\lambda}_1(t)$")
    ax.set_title("最大 Lyapunov 指数收敛")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mle_convergence.png"), dpi=300)
    plt.close(fig)


def plot_integrator_crosscheck(
    t: np.ndarray,
    x_rk: np.ndarray,
    x_ra: np.ndarray,
    ev_rk: np.ndarray,
    ev_ra: np.ndarray,
    outdir: str,
) -> None:
    """绘制 RK45 与 Radau 积分器的交叉验证图。
    
    生成包含两个子图的图表：
    - 上图：两种积分器的轨迹对比，并标记 Poincaré 事件时间
    - 下图：两种积分器的轨迹差值
    
    参数:
        t: 时间数组
        x_rk: RK45 的 x 分量时间序列
        x_ra: Radau 的 x 分量时间序列
        ev_rk: RK45 的 Poincaré 事件时间数组
        ev_ra: Radau 的 Poincaré 事件时间数组
        outdir: 输出目录
        
    输出文件:
        integrator_crosscheck.png
    """
    _ensure_dir(outdir)
    fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.8), sharex=True)

    # 轨迹对比
    axs[0].plot(t, x_rk, lw=1.0, label="RK45")
    axs[0].plot(t, x_ra, lw=1.0, ls="--", label="Radau")
    axs[0].set_ylabel("$x$")
    axs[0].set_title("积分器交叉验证：轨迹对比")
    axs[0].legend()

    # 可选：标记 Poincaré 事件时间
    for te in ev_rk:
        axs[0].axvline(te, color="C0", alpha=0.2, lw=0.8)
    for te in ev_ra:
        axs[0].axvline(te, color="C1", alpha=0.2, lw=0.8)

    # 轨迹差值
    axs[1].plot(t, x_rk - x_ra, color="C3", lw=1.0)
    axs[1].set_xlabel("时间 $t$")
    axs[1].set_ylabel("$x_{\\mathrm{RK45}}-x_{\\mathrm{Radau}}$")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "integrator_crosscheck.png"), dpi=300)
    plt.close(fig)


def plot_acf(lags: np.ndarray, acf: np.ndarray, outdir: str) -> None:
    """绘制自相关函数（ACF）。
    
    绘制时间序列的自相关函数，展示信号在不同滞后下的相关性。
    
    参数:
        lags: 滞后数组
        acf: 自相关函数值数组
        outdir: 输出目录
        
    输出文件:
        acf_x.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.8))
    ax.plot(lags, acf, lw=1.0)
    ax.set_xlabel("滞后 $k$")
    ax.set_ylabel("ACF$(k)$")
    ax.set_title("自相关函数 (ACF)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "acf_x.png"), dpi=300)
    plt.close(fig)


def plot_histogram(
    data: np.ndarray,
    bins: int = 64,
    outdir: str = "output",
    fname: str = "hist.png",
    xlabel: str = "x",
    title: str = "Histogram",
) -> None:
    """通用直方图绘制函数。
    
    绘制数据的直方图，显示概率密度分布。
    
    参数:
        data: 输入数据数组
        bins: 直方图区间数（默认 64）
        outdir: 输出目录（默认 "output"）
        fname: 输出文件名（默认 "hist.png"）
        xlabel: 横轴标签（默认 "x"）
        title: 图表标题（默认 "Histogram"）
        
    默认输出:
        output/hist.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    ax.hist(data, bins=bins, density=True, color="C0", alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("概率密度")
    ax.set_title(title)
    fig.tight_layout()
    ax.figure.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close(fig)


def plot_poincare_scatter(px: np.ndarray, pz: np.ndarray, outdir: str) -> None:
    """绘制 Poincaré 截面散点图 (x, z)。
    
    绘制 Poincaré 截面上的点分布，展示吸引子在截面上的几何结构。
    横轴为 x 坐标，纵轴为 z 坐标。
    
    参数:
        px: Poincaré 截面上的 x 坐标数组
        pz: Poincaré 截面上的 z 坐标数组
        outdir: 输出目录
        
    输出文件:
        poincare_scatter.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    ax.scatter(px, pz, s=5, alpha=0.65)
    ax.set_xlabel("截面坐标 $x$")
    ax.set_ylabel("截面坐标 $z$")
    ax.set_title("Poincaré 截面 ($y=0$)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "poincare_scatter.png"), dpi=300)
    plt.close(fig)


def plot_mle_hist(lambdas: np.ndarray, outdir: str) -> None:
    """绘制蒙特卡洛 MLE 样本的直方图。
    
    绘制在参数扰动下进行蒙特卡洛分析得到的 MLE 样本分布直方图，
    用于评估 MLE 对参数扰动的敏感性。
    
    参数:
        lambdas: MLE 样本数组
        outdir: 输出目录
        
    输出文件:
        mle_mc_hist.png
    """
    _ensure_dir(outdir)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    ax.hist(lambdas, bins=30, color="C0", alpha=0.75)
    ax.set_xlabel("$\\lambda_1$")
    ax.set_ylabel("频数")
    ax.set_title("最大 Lyapunov 指数的蒙特卡洛分布")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mle_mc_hist.png"), dpi=300)
    plt.close(fig)
