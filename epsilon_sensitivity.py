#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描平滑参数 epsilon 对最大 Lyapunov 指数的影响，并生成敏感性图。

输出：
- output/epsilon_mle_sensitivity.png
- output/epsilon_mle_sensitivity.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chua_py.config import ChuaParams
from chua_py.lyapunov import compute_mle

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Songti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    """主函数：执行平滑参数敏感性分析。
    
    扫描平滑参数 epsilon 在 [10^-7, 10^-5] 区间内对最大 Lyapunov 指数的影响。
    对每个 epsilon 值计算 MLE，并生成敏感性曲线图和 CSV 数据文件。
    """
    base = Path(__file__).resolve().parent
    outdir = base / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    # 扫描平滑参数 epsilon（对 eps_smooth）
    # 在对数刻度上均匀分布 7 个点，范围从 10^-7 到 10^-5
    eps_vals = np.logspace(-7, -5, 7)
    mle_vals = []

    p0 = ChuaParams()  # 基础参数配置
    for eps in eps_vals:
        # 为每个 epsilon 值创建新的参数对象
        p = ChuaParams(**p0.__dict__)
        p.eps_smooth = float(eps)
        # 计算该 epsilon 值对应的 MLE
        lam, _, _ = compute_mle(p)
        mle_vals.append(lam)

    eps_vals = np.asarray(eps_vals)
    mle_vals = np.asarray(mle_vals)
    # 找到最接近默认值 p0.eps_smooth 的参考值
    mle_ref = mle_vals[np.argmin(np.abs(eps_vals - p0.eps_smooth))]
    # 计算相对误差
    rel_err = (mle_vals - mle_ref) / np.abs(mle_ref)

    df = pd.DataFrame(
        {
            "epsilon": eps_vals,
            "lambda1": mle_vals,
            "rel_error": rel_err,
        }
    )
    df.to_csv(outdir / "epsilon_mle_sensitivity.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.semilogx(eps_vals, mle_vals, "o-", label=r"$\lambda_1(\epsilon)$")
    plt.axhline(
        mle_ref,
        color="k",
        linestyle="--",
        linewidth=1.0,
        label=rf"参考值 $\epsilon={p0.eps_smooth:g}$",
    )
    plt.xlabel(r"平滑参数 $\epsilon$")
    plt.ylabel(r"最大 Lyapunov 指数 $\lambda_1$")
    plt.title(r"平滑参数 $\epsilon$ 对最大 Lyapunov 指数的影响")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "epsilon_mle_sensitivity.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

