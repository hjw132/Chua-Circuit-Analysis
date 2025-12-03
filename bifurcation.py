from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable

from .config import ChuaParams
from .integrate import integrate_trajectory


def _eval_states_at(sol, t_events: np.ndarray) -> np.ndarray:
	"""在事件时间点评估状态。
	
	辅助函数，用于在 Poincaré 事件时间点获取状态值。
	
	参数:
		sol: SciPy 的 OdeResult 对象
		t_events: 事件时间数组
		
	返回:
		状态数组，形状为 (n_events, 3)
	"""
	if getattr(sol, "sol", None) is None:
		# 回退方案：从记录的积分点进行线性插值
		y = np.vstack([np.interp(t_events, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]).T
	else:
		# 使用密集输出插值函数（更精确）
		y = sol.sol(t_events).T
	return y


def scan_parameter(
	p: ChuaParams,
	param: str = "m1",
	values: Iterable[float] | None = None,
	t_burn: float | None = None,
	t_window: float = 200.0,
	num_starts: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
	"""扫描参数并收集 Poincaré 截面上的 x 值，用于生成分岔图。
	
	在指定参数范围内扫描，对每个参数值：
	1. 进行预热积分
	2. 在采样窗口内积分并检测 Poincaré 事件
	3. 收集截面上的 x 坐标值
	
	支持多起始点（num_starts > 1）以捕获多个吸引子或提高统计可靠性。
	使用路径跟随策略：每个参数值的初始条件基于上一个参数值的最终状态。
	
	参数:
		p: 参数对象（会被修改）
		param: 要扫描的参数名称（默认 "m1"）
		values: 参数值列表（None 时使用 p.bifur_min 到 p.bifur_max 的均匀网格）
		t_burn: 预热时间（默认 p.t_burn）
		t_window: 采样窗口长度（默认 200.0）
		num_starts: 每个参数值的随机起始点数（默认 1）
		
	返回:
		(param_values_repeated, x_on_section): 元组
		- param_values_repeated: 参数值数组（每个事件对应一个值）
		- x_on_section: Poincaré 截面上的 x 坐标数组
	"""
	if t_burn is None:
		t_burn = p.t_burn

	if values is None:
		# 使用默认参数范围生成均匀网格
		values = np.linspace(p.bifur_min, p.bifur_max, p.bifur_grid)

	rng = np.random.default_rng(p.seed)
	start = rng.normal(scale=0.1, size=3)  # 初始锚点
	param_vals = []
	section_x = []

	for val in values:
		# 设置参数值
		if not hasattr(p, param):
			raise AttributeError(f"未知参数: {param}")
		setattr(p, param, float(val))

		for k in range(max(1, int(num_starts))):
			# 为每个起始点，在上一个锚点附近生成新的初始条件
			start_k = start + 0.05 * rng.normal(size=3)
			
			# 预热阶段
			res_burn = integrate_trajectory(p, start_k, (0.0, float(t_burn)), with_events=False)
			if not res_burn.success:
				continue
			state_after = res_burn.y[:, -1]
			
			# 采样窗口：启用 Poincaré 事件检测
			res = integrate_trajectory(
				p,
				state_after,
				(float(t_burn), float(t_burn + t_window)),
				with_events=True,
				dense_output=True,
			)
			if not res.success:
				continue
			if len(res.t_events) == 0 or len(res.t_events[0]) == 0:
				continue
			
			# 提取事件时间和对应的状态
			ts = res.t_events[0]
			y_on_events = _eval_states_at(res, ts)
			x_values = y_on_events[:, 0]  # 提取 x 坐标
			
			# 保存数据
			param_vals.append(np.full_like(x_values, fill_value=val, dtype=float))
			section_x.append(x_values)
			
			# 更新锚点：使用最后一次成功运行的最终状态（路径跟随策略）
			start = res.y[:, -1]

	if len(param_vals) == 0:
		return np.array([]), np.array([])

	return np.concatenate(param_vals), np.concatenate(section_x)


