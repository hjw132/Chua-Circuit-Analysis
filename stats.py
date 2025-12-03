from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_acf(x: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
	"""计算自相关函数，直到最大滞后 max_lag（包含）。
	
	自相关函数衡量时间序列与其自身在不同时间滞后下的相关性。
	对于滞后 k，ACF(k) = E[(x_t - μ)(x_{t+k} - μ)] / σ²，其中 μ 是均值，σ² 是方差。
	
	参数:
		x: 输入时间序列，一维数组
		max_lag: 最大滞后点数（包含）
		
	返回:
		(lags, acf): 元组
		- lags: 滞后数组 [0, 1, 2, ..., max_lag]
		- acf: 自相关函数值数组，acf[0] == 1（零滞后时完全相关）
	"""
	x = np.asarray(x, dtype=float)
	x = x - x.mean()  # 去均值
	n = len(x)
	# 对于大数据集，可以使用 FFT 卷积加速
	# 这里使用简单方法以提高可读性（因为 n 通常不大）
	acf = np.empty(max_lag + 1, dtype=float)
	var = np.dot(x, x) / n  # 计算方差
	if var <= 0.0:
		# 方差为零（常数序列）的特殊情况
		acf[:] = 0.0
		acf[0] = 1.0
		return np.arange(max_lag + 1), acf
	for k in range(max_lag + 1):
		# 计算滞后 k 的自相关：C(k) = <x[t] * x[t+k]> / (n-k) / var
		acf[k] = np.dot(x[: n - k], x[k: n]) / ((n - k) * var)
	return np.arange(max_lag + 1), acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
	"""估计积分自相关时间 tau_int。
	
	积分自相关时间衡量时间序列的有效独立样本数。如果原始序列有 N 个样本，
	则有效独立样本数约为 N / tau_int。
	
	使用正滞后直到第一次非正穿越来计算：
		tau_int = 1 + 2 * sum_{k=1..K} acf[k]
	其中 K 是第一个满足 acf[k] <= 0 的索引（不包含）。
	
	参数:
		acf: 自相关函数数组，acf[0] 应该为 1
		
	返回:
		积分自相关时间（浮点数）
	"""
	if len(acf) <= 1:
		return 0.0
	# 找到滞后 0 之后第一个非正穿越点
	pos = np.where(acf[1:] <= 0.0)[0]
	K = int(pos[0]) if pos.size > 0 else len(acf) - 1
	return float(1.0 + 2.0 * np.sum(acf[1 : K + 1]))


def shannon_entropy(data: np.ndarray, bins: int = 64) -> Tuple[float, np.ndarray, np.ndarray]:
	"""通过直方图计算一维数据的 Shannon 熵（自然对数）。
	
	Shannon 熵衡量数据的随机性或信息量。对于 Poincaré 截面数据，熵值越高
	表示截面上的点分布越均匀，系统的混合性越强。
	
	公式：H = -sum(p_i * log(p_i))，其中 p_i 是第 i 个区间的概率。
	
	参数:
		data: 输入数据，一维数组
		bins: 直方图的区间数（默认 64）
		
	返回:
		(entropy, hist, edges): 元组
		- entropy: Shannon 熵值（单位：nats，自然对数）
		- hist: 直方图计数值数组
		- edges: 直方图边界数组
	"""
	data = np.asarray(data, dtype=float)
	hist, edges = np.histogram(data, bins=bins, density=True)
	# 计算每个区间的概率
	p = hist * np.diff(edges)
	# 只考虑非零概率（避免 log(0)）
	p = p[p > 0]
	# 计算 Shannon 熵
	H = -np.sum(p * np.log(p))
	return float(H), hist, edges



