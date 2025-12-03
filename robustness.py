from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple

from .config import ChuaParams
from .lyapunov import compute_mle


def _draw_params(p: ChuaParams, sigma: float, rng: np.random.Generator) -> ChuaParams:
	"""生成参数对象的浅拷贝，并对关键参数施加乘性高斯扰动。
	
	这是蒙特卡洛鲁棒性分析的辅助函数。对每个关键参数（alpha, beta, rho, m0, m1）
	施加相对标准差为 sigma 的高斯扰动。
	
	参数:
		p: 原始参数对象
		sigma: 扰动的相对标准差
		rng: 随机数生成器
		
	返回:
		扰动后的参数对象（浅拷贝）
	"""
	q = ChuaParams(**p.__dict__)
	for name in ["alpha", "beta", "rho", "m0", "m1"]:
		val = getattr(p, name)
		# 乘性扰动：新值 = 原值 * (1 + sigma * N(0,1))
		pert = float(val) * (1.0 + sigma * rng.normal())
		setattr(q, name, pert)
	return q


def monte_carlo_mle(
	p: ChuaParams,
	n_samples: int,
	sigma: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
	"""在参数扰动下进行 MLE 的蒙特卡洛分析。
	
	评估最大 Lyapunov 指数对参数扰动的敏感性。对每个样本：
	1. 对关键参数施加随机扰动
	2. 计算扰动后的 MLE
	3. 统计 MLE 的分布特性
	
	参数:
		p: 基础参数对象
		n_samples: 蒙特卡洛样本数
		sigma: 参数扰动的相对标准差
		
	返回:
		(lambda1_samples, summary_dict): 元组
		- lambda1_samples: MLE 样本数组
		- summary_dict: 统计摘要字典，包含均值、标准差、分位数等
	"""
	rng = np.random.default_rng(p.seed)
	samples = []
	for i in range(int(n_samples)):
		# 生成扰动后的参数
		pi = _draw_params(p, sigma, rng)
		try:
			# 计算扰动后的 MLE
			lambda1, _, _ = compute_mle(pi)
			samples.append(lambda1)
		except Exception:
			# 跳过失败的运行（可能由于数值不稳定）
			continue

	arr = np.asarray(samples, dtype=float)
	if arr.size == 0:
		return arr, {"count": 0}

	# 计算统计摘要
	summary = {
		"count": int(arr.size),                    # 成功样本数
		"mean": float(np.mean(arr)),               # 均值
		"std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,  # 标准差
		"min": float(np.min(arr)),                 # 最小值
		"max": float(np.max(arr)),                 # 最大值
		"p05": float(np.quantile(arr, 0.05)),      # 5% 分位数
		"p50": float(np.quantile(arr, 0.50)),      # 中位数（50% 分位数）
		"p95": float(np.quantile(arr, 0.95)),      # 95% 分位数
	}
	return arr, summary



