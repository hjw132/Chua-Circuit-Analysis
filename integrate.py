from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional, Callable

from .config import ChuaParams
from .model import rhs, poincare_event


def integrate_trajectory(
	p: ChuaParams,
	x0: np.ndarray,
	t_span: Tuple[float, float],
	method: Optional[str] = None,
	with_events: bool = True,
	dense_output: bool = True,
	max_step: Optional[float] = None,
	event_fun: Optional[Callable] = None,
):
	"""使用 SciPy 的 solve_ivp 积分轨迹。
	
	默认安装 Poincaré 截面事件（y=0，方向+1，向上穿越）。
	该函数封装了 SciPy 的 solve_ivp，提供了便捷的参数接口。
	
	参数:
		p: 参数对象
		x0: 初始状态 [x0, y0, z0]
		t_span: 积分时间区间 (t_start, t_end)
		method: 积分方法（None 时使用 p.method，默认 "RK45"）
		with_events: 是否启用事件检测（默认 True）
		dense_output: 是否生成密集输出（用于后续采样，默认 True）
		max_step: 最大步长限制（None 时使用 SciPy 默认值）
		event_fun: 自定义事件函数（None 时使用默认 Poincaré 事件）
		
	返回:
		SciPy 的 OdeResult 对象，包含：
		- t: 时间点数组
		- y: 状态数组（shape: (3, n_points)）
		- sol: 密集输出插值函数（如果 dense_output=True）
		- t_events: 事件时间列表
	"""
	if method is None:
		method = p.method

	if with_events and event_fun is None:
		# 提供默认事件函数：Poincaré 截面（y=0，向上穿越）
		# 注意：SciPy 要求事件函数签名为 (t, y) -> float，但我们需要参数 p
		def _ev(t, y):
			return poincare_event(t, y, p)

		_ev.direction = 1.0  # type: ignore[attr-defined]
		"""事件方向：+1 表示仅检测向上穿越"""
		
		_ev.terminal = False  # type: ignore[attr-defined]
		"""非终止事件：积分继续进行"""
		
		event_fun = _ev

	res = solve_ivp(
		fun=lambda t, y: rhs(t, y, p),  # 状态方程右端项
		y0=np.asarray(x0, dtype=float),  # 初始状态
		t_span=t_span,                   # 时间区间
		method=method,                  # 积分方法（RK45 或 Radau）
		rtol=p.rtol,                    # 相对容差（默认 1e-9）
		atol=p.atol,                    # 绝对容差（默认 1e-12）
		dense_output=dense_output,       # 密集输出
		events=event_fun if with_events else None,  # 事件检测
		# 仅在明确提供时传递 max_step；否则使用 SciPy 默认值（np.inf）
		**({"max_step": max_step} if max_step is not None else {}),
	)
	return res


def sample_dense(sol, t0: float, t1: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
	"""从密集输出中均匀采样，返回时间序列和状态序列。
	
	在指定时间区间 [t0, t1] 上以固定间隔 dt 均匀采样状态。
	如果积分器提供了密集输出插值函数，则使用它；否则回退到线性插值。
	
	参数:
		sol: SciPy 的 OdeResult 对象（积分结果）
		t0: 采样起始时间
		t1: 采样结束时间
		dt: 采样时间间隔
		
	返回:
		(times, states): 元组
		- times: 时间数组，形状为 (N,)
		- states: 状态数组，形状为 (N, 3)，每行是一个状态 [x, y, z]
	"""
	if not sol.success:
		raise RuntimeError("积分器失败: " + str(sol.message))

	# 生成均匀时间网格
	times = np.arange(t0, t1 + 1e-12, dt)
	
	if getattr(sol, "sol", None) is None:
		# 回退方案：从记录的积分步进行线性插值
		states = np.vstack([
			np.interp(times, sol.t, sol.y[i]) for i in range(sol.y.shape[0])
		]).T
	else:
		# 使用密集输出插值函数（更精确）
		states = sol.sol(times).T  # 形状 (N, 3)
	return times, states


def kcl_residual(times: np.ndarray, states: np.ndarray, p: ChuaParams) -> Tuple[np.ndarray, np.ndarray]:
	"""计算基于有限差分的物理一致性残差。
	
	通过比较数值导数和状态方程右端项来验证数值解的物理一致性。
	对于每个采样时间点，使用前向/后向/中心差分近似 ds/dt，然后与右端项
	进行比较，返回无穷范数残差。
	
	该残差可用于诊断数值积分的精度，特别是在切换点附近。
	
	参数:
		times: 时间数组
		states: 状态数组，形状为 (N, 3)
		p: 参数对象
		
	返回:
		(times, residual): 元组
		- times: 时间数组（与输入相同）
		- residual: 残差数组，每个元素是该时间点的最大残差（无穷范数）
	"""
	N = len(times)
	residual = np.zeros(N)
	for i in range(N):
		# 使用有限差分近似数值导数
		if i == 0:
			# 第一个点：前向差分
			dsdt = (states[i + 1] - states[i]) / (times[i + 1] - times[i])
		elif i == N - 1:
			# 最后一个点：后向差分
			dsdt = (states[i] - states[i - 1]) / (times[i] - times[i - 1])
		else:
			# 中间点：中心差分（更精确）
			dsdt = (states[i + 1] - states[i - 1]) / (times[i + 1] - times[i - 1])
		
		# 计算状态方程右端项
		rhs_i = rhs(times[i], states[i], p)
		
		# 计算残差的无穷范数（三个分量中的最大绝对误差）
		residual[i] = np.max(np.abs(dsdt - rhs_i))
	return times, residual


