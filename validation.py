from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any

from .config import ChuaParams
from .integrate import integrate_trajectory, sample_dense, kcl_residual


def cross_check_integrators(
	p: ChuaParams,
	x0: np.ndarray,
	t_span: Tuple[float, float],
	dt_sample: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""使用相同设置运行 RK45 和 Radau 积分器，并比较诊断结果。
	
	这是数值验证框架的核心函数，通过对比两种不同积分器的结果来评估
	数值解的可靠性。RK45 是显式方法，Radau 是隐式方法，两者在相同容差
	下应该给出相似的结果。
	
	参数:
		p: 参数对象
		x0: 初始状态
		t_span: 积分时间区间 (t_start, t_end)（预热已在调用者处处理）
		dt_sample: 采样时间间隔
		
	返回:
		(metrics, times, x_rk, x_radau, (ev_rk, ev_ra)): 元组
		- metrics: 诊断指标字典，包含容差、残差、状态差异等
		- times: 采样时间数组
		- x_rk: RK45 的 x 分量时间序列
		- x_radau: Radau 的 x 分量时间序列
		- (ev_rk, ev_ra): 元组，包含两个积分器的 Poincaré 事件时间数组
	"""
	# 使用 RK45（显式 Runge-Kutta 方法）积分
	res_rk = integrate_trajectory(
		p,
		x0,
		t_span,
		method="RK45",
		with_events=True,
		dense_output=True,
		max_step=p.max_step,
	)
	if not res_rk.success:
		raise RuntimeError("RK45 积分失败: " + str(res_rk.message))
	
	# 使用 Radau（隐式 Radau IIA 方法）积分
	res_ra = integrate_trajectory(
		p,
		x0,
		t_span,
		method="Radau",
		with_events=True,
		dense_output=True,
		max_step=p.max_step,
	)
	if not res_ra.success:
		raise RuntimeError("Radau 积分失败: " + str(res_ra.message))

	# 共享采样网格（预热已在调用者的 t_span 中处理）
	t0, t1 = t_span
	times = np.arange(t0, t1 + 1e-12, dt_sample)
	tr_rk, sr_rk = sample_dense(res_rk, t0, t1, dt_sample)
	tr_ra, sr_ra = sample_dense(res_ra, t0, t1, dt_sample)
	# 预期网格相同
	assert np.allclose(tr_rk, times) and np.allclose(tr_ra, times)

	# 计算 KCL 残差（物理一致性检查）
	_, r_rk = kcl_residual(tr_rk, sr_rk, p)
	_, r_ra = kcl_residual(tr_ra, sr_ra, p)

	# 提取 Poincaré 事件时间
	ev_rk = res_rk.t_events[0] if len(res_rk.t_events) > 0 else np.array([])
	ev_ra = res_ra.t_events[0] if len(res_ra.t_events) > 0 else np.array([])

	# 按索引比较事件时间差异（取较短的长度）
	n_match = int(min(len(ev_rk), len(ev_ra)))
	if n_match > 0:
		ev_diff = np.max(np.abs(ev_rk[:n_match] - ev_ra[:n_match]))
	else:
		ev_diff = float("nan")

	# 在采样网格上计算状态的无穷范数差异
	state_diff_sup = float(np.max(np.abs(sr_rk - sr_ra)))

	# 汇总诊断指标
	metrics = {
		"rtol": p.rtol,
		"atol": p.atol,
		"max_step": p.max_step,
		"residual_inf_rk_max": float(np.max(r_rk)),      # RK45 最大残差
		"residual_inf_radau_max": float(np.max(r_ra)),   # Radau 最大残差
		"state_diff_sup": state_diff_sup,                 # 状态最大差异
		"events_count_rk": int(len(ev_rk)),              # RK45 事件数
		"events_count_radau": int(len(ev_ra)),           # Radau 事件数
		"event_time_max_abs_diff": float(ev_diff),       # 事件时间最大差异
	}

	return metrics, times, sr_rk[:, 0], sr_ra[:, 0], (ev_rk, ev_ra)



