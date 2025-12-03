from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

from chua_py import (
    ChuaParams,
    integrate_trajectory,
    sample_dense,
    kcl_residual,
    scan_parameter,
    compute_mle,
    cross_check_integrators,
    compute_acf,
    integrated_autocorrelation_time,
    shannon_entropy,
    monte_carlo_mle,
)
from chua_py.plotting import (
    plot_time_phase,
    plot_residual,
    plot_bifurcation,
    plot_mle,
    plot_integrator_crosscheck,
    plot_histogram,
    plot_poincare_scatter,
    plot_mle_hist,
    plot_circuit_schematic_sd,
    plot_state_space_block_sd,
    plot_attractor_3d,
    plot_stats_panels,
    plot_validation_panels,
    compute_psd_series,
)


def ensure_outdir() -> str:
    """确保输出目录存在并返回其路径。
    
    如果 output/ 目录不存在，则创建它。返回目录的字符串路径。
    
    返回:
        输出目录的路径字符串
    """
    base = Path(__file__).resolve().parent
    out = base / "output"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def run_baseline() -> None:
    """运行基线仿真并生成所有图表和数据文件。
    
    这是主实验脚本，执行完整的数值仿真流程，包括：
    1. 单条轨迹积分与相图生成
    2. 最大 Lyapunov 指数计算
    3. 积分器交叉验证（RK45 vs Radau）
    4. 参数 m1 的一维分岔图扫描
    5. Poincaré 截面分析与 Shannon 熵计算
    6. 功率谱密度与自相关函数分析
    7. KCL 残差诊断
    8. 蒙特卡洛鲁棒性分析（如果启用）
    
    所有结果保存至 output/ 目录，包括 CSV 数据文件、JSON 元数据以及 PNG/PDF 图表。
    """
    # 初始化参数和随机数生成器
    p = ChuaParams()
    rng = np.random.default_rng(p.seed)
    # 生成随机初始状态（小幅度扰动）
    x0 = rng.normal(scale=0.1, size=3)

    fig_dir = Path(__file__).resolve().parent
    outdir = Path(ensure_outdir())
    # 保存参数配置到 JSON 文件
    with open(outdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(p.__dict__, f, indent=2)

    # 生成电路原理图和状态空间方块图（使用 schemdraw，可选）
    try:
        plot_circuit_schematic_sd(str(outdir))
        plot_state_space_block_sd(str(outdir), p)
        print("schemdraw 电路图和方块图已生成。")
    except Exception as e:  # pragma: no cover - optional convenience
        print(f"schemdraw 图表生成失败（已忽略）: {e}")

    # ========== 1) 单条轨迹积分 ==========
    res = integrate_trajectory(
        p,
        x0,
        (0.0, p.t_end),
        with_events=True,      # 启用 Poincaré 事件检测
        dense_output=True,     # 生成密集输出用于后续采样
        max_step=p.max_step,
    )
    if not res.success:
        raise SystemExit("积分失败: " + str(res.message))

    # 预热后均匀采样
    t, s = sample_dense(res, p.t_burn, p.t_end, p.dt_save)
    # 绘制时间序列和相图
    plot_time_phase(t, s, fig_dir)
    # 绘制三维吸引子
    plot_attractor_3d(s, fig_dir)

    # 保存轨迹数据到 CSV
    df = pd.DataFrame({"t": t, "x": s[:, 0], "y": s[:, 1], "z": s[:, 2]})
    df.to_csv(outdir / "trajectory.csv", index=False)

    # ========== PSD + ACF（合并的统计面板）==========
    # 计算功率谱密度
    freq, psd = compute_psd_series(t, s[:, 0])
    # 计算自相关函数
    lags, acf = compute_acf(s[:, 0], max_lag=p.acf_max_lag)
    # 绘制统计面板（PSD 和 ACF）
    plot_stats_panels(freq, psd, lags, acf, fig_dir)
    # 保存 ACF 相关统计量
    with open(outdir / "acf.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_lag": int(p.acf_max_lag),
                "tau_int": integrated_autocorrelation_time(acf),
            },
            f,
            indent=2,
        )

    # ========== 残差诊断 ==========
    # 计算基于 KCL 的物理一致性残差
    tr, r = kcl_residual(t, s, p)
    plot_residual(tr, r, fig_dir)
    # 保存残差数据
    pd.DataFrame({"t": tr, "residual_inf": r}).to_csv(outdir / "residual.csv", index=False)

    # ========== Poincaré 截面分析（来自基线运行）==========
    if len(res.t_events) > 0 and len(res.t_events[0]) > 0:
        te = res.t_events[0]  # Poincaré 事件时间
        # 在事件时间点评估状态
        if getattr(res, "sol", None) is not None:
            sec_states = res.sol(te).T
        else:
            # 回退到线性插值
            sec_states = np.vstack(
                [np.interp(te, res.t, res.y[i]) for i in range(res.y.shape[0])]
            ).T
        px, pz = sec_states[:, 0], sec_states[:, 2]  # 提取 x 和 z 坐标
        # 绘制 Poincaré 截面散点图
        plot_poincare_scatter(px, pz, str(outdir))
        # 保存截面数据
        pd.DataFrame({"t": te, "x": px, "z": pz}).to_csv(outdir / "poincare_section.csv", index=False)
        # 计算 Shannon 熵
        H, hist, edges = shannon_entropy(px, bins=64)
        # 绘制截面 x 坐标的直方图
        plot_histogram(
            px,
            bins=64,
            outdir=str(outdir),
            fname="poincare_x_hist.png",
            xlabel="截面坐标 $x$",
            title="Poincaré 截面上 $x$ 坐标的直方图",
        )
        # 保存熵值
        with open(
            outdir / "poincare_entropy.json", "w", encoding="utf-8"
        ) as f:
            json.dump({"entropy_nats": float(H)}, f, indent=2)

    # ========== 2) Lyapunov 最大指数（MLE）计算 ==========
    l1, times, running = compute_mle(p, x0=x0)
    # 绘制 MLE 收敛曲线
    plot_mle(times, running, str(outdir))
    # 保存 MLE 结果
    with open(outdir / "mle.json", "w", encoding="utf-8") as f:
        json.dump({"lambda1": float(l1)}, f, indent=2)
    print(f"MLE (lambda1) ~ {l1:.5f}")

    # ========== 2.5) 在采样窗口上交叉验证 RK45 vs Radau ==========
    metrics, tc, xrk, xra, (ev_rk, ev_ra) = cross_check_integrators(
        p, x0, (p.t_burn, p.t_end), p.dt_save
    )
    # 绘制积分器交叉验证图
    plot_integrator_crosscheck(tc, xrk, xra, ev_rk, ev_ra, str(outdir))
    # 绘制验证面板（轨迹对比和残差）
    plot_validation_panels(tc, xrk, xra, ev_rk, ev_ra, tr, r, fig_dir)
    # 保存交叉验证指标
    with open(
        outdir / "integrator_crosscheck.json", "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    # ========== 3) 参数 m1 的分岔图扫描（多起始点）==========
    vals = np.linspace(p.bifur_min, p.bifur_max, p.bifur_grid)
    params_rep, x_on_sec = scan_parameter(
        p,
        param=p.bifur_param,
        values=vals,
        t_burn=p.t_burn,
        t_window=200.0,
        num_starts=p.bifur_starts,
    )
    if params_rep.size > 0:
        # 绘制分岔图
        plot_bifurcation(
            params_rep,
            x_on_sec,
            str(outdir),
            xlabel="参数 $m_1'",
        )
        # 保存分岔数据
        pd.DataFrame({"param": params_rep, "x_section": x_on_sec}).to_csv(
            outdir / "bifurcation.csv", index=False
        )
    else:
        print("警告：未收集到 Poincaré 事件，无法生成分岔图。")

    # ========== 4) MLE 的蒙特卡洛鲁棒性分析（如果启用）==========
    if p.mc_samples and p.mc_samples > 0:
        arr, summary = monte_carlo_mle(p, n_samples=p.mc_samples, sigma=p.param_sigma)
        if arr.size > 0:
            # 绘制 MLE 分布直方图
            plot_mle_hist(arr, str(outdir))
            # 保存 MLE 样本
            pd.DataFrame({"lambda1": arr}).to_csv(
                outdir / "mle_mc.csv", index=False
            )
        # 保存统计摘要
        with open(
            outdir / "mle_mc_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

    print("所有结果已保存至:", outdir)


if __name__ == "__main__":
    run_baseline()
