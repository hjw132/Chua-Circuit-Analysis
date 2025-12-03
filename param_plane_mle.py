#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在二维参数平面上扫描最大 Lyapunov 指数示意图。

示例：以 (alpha, m1) 为横纵坐标，颜色表示 lambda1。

输出：
- output/mle_param_plane_alpha_m1.png
- output/mle_param_plane_alpha_m1.npy (包含 alpha_grid, m1_grid, lambda1_grid)
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from chua_py.config import ChuaParams
from chua_py.lyapunov import compute_mle

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    """主函数：在二维参数平面上扫描最大 Lyapunov 指数。
    
    在 (alpha, m1) 参数平面上进行网格扫描，计算每个参数组合对应的 MLE，
    并生成热力图展示 MLE 的分布。颜色表示 lambda1 的数值：
    - 暖色对应较大正值（混沌区域）
    - 冷色对应接近零或负值（非混沌区域）
    """
    base = Path(__file__).resolve().parent
    outdir = base / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    # 粗略扫描参数平面：alpha x m1
    # alpha 在 [8.0, 12.0] 范围内均匀取 15 个点
    alpha_vals = np.linspace(8.0, 12.0, 15)
    # m1 在 [-0.80, -0.55] 范围内均匀取 15 个点
    m1_vals = np.linspace(-0.80, -0.55, 15)

    # 初始化 MLE 网格（15x15），初始值为 NaN
    lam_grid = np.full((len(m1_vals), len(alpha_vals)), np.nan, dtype=float)

    p_base = ChuaParams()
    # 为避免计算量过大，缩短 MLE 积分时间（从默认 600.0 缩短到 150.0）
    mle_total_original = p_base.mle_total
    p_base.mle_total = 150.0

    # 双重循环遍历参数网格
    for i, m1 in enumerate(m1_vals):
        for j, alpha in enumerate(alpha_vals):
            # 为每个参数组合创建新的参数对象
            p = ChuaParams(**p_base.__dict__)
            p.alpha = float(alpha)
            p.m1 = float(m1)
            try:
                # 计算该参数组合的 MLE
                lam, _, _ = compute_mle(p)
                lam_grid[i, j] = lam
            except Exception:
                # 如果计算失败（可能由于数值不稳定），标记为 NaN
                lam_grid[i, j] = np.nan

    np.save(
        outdir / "mle_param_plane_alpha_m1.npy",
        {"alpha": alpha_vals, "m1": m1_vals, "lambda1": lam_grid},
        allow_pickle=True,
    )
    p_base.mle_total = mle_total_original

    A, M1 = np.meshgrid(alpha_vals, m1_vals)
    plt.figure(figsize=(6, 4.5))
    cmap = plt.get_cmap("viridis")
    im = plt.pcolormesh(A, M1, lam_grid, shading="auto", cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.set_label(r"最大 Lyapunov 指数 $\lambda_1$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$m_1$")
    plt.title(r"$(\alpha,m_1)$ 参数平面上最大 Lyapunov 指数分布")
    plt.tight_layout()
    plt.savefig(outdir / "mle_param_plane_alpha_m1.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
