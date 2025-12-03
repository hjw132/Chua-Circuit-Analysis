from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple

from .config import ChuaParams
from .model import rhs, jacobian


def _extended_rhs(t: float, y: np.ndarray, p: ChuaParams) -> np.ndarray:
	"""扩展系统右端项（用于 MLE 计算）。
	
	扩展系统包含状态 x(3维) 和切向量矩阵 Phi(3×3=9维)，总维度为 12。
	
	扩展系统方程：
		d/dt [x; Phi] = [f(x); J_ε(x) * Phi]
	
	其中 J_ε(x) 是平滑雅可比矩阵，用于提高变分方程的数值稳定性。
	
	参数:
		t: 时间
		y: 扩展状态 [x(3), Phi(9)]，Phi 按行展开为一维数组
		p: 参数对象
		
	返回:
		扩展状态导数 [dx/dt(3), dPhi/dt(9)]，一维数组
	"""
	# y = [x(3), Phi(9)]，总长度 12
	s = y[:3]                    # 提取状态 x
	phi = y[3:].reshape(3, 3)   # 提取切向量矩阵 Phi (3×3)
	
	ds = rhs(t, s, p)            # 状态导数：dx/dt = f(x)
	J = jacobian(t, s, p, smooth=True)  # 平滑雅可比矩阵
	dphi = J @ phi               # 切向量导数：dPhi/dt = J * Phi
	
	return np.hstack([ds, dphi.ravel()])  # 合并为 12 维向量


def _propagate_segment(p: ChuaParams, s0: np.ndarray, phi0: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
	"""在短时间区间 [t0, t1] 上传播状态和切向量。
	
	这是 MLE 计算中的核心步骤，在每个 QR 重正交间隔内积分扩展系统。
	
	参数:
		p: 参数对象
		s0: 初始状态 [x0, y0, z0]
		phi0: 初始切向量矩阵 (3×3)
		t0: 积分起始时间
		t1: 积分结束时间
		
	返回:
		(s1, phi1): 元组
		- s1: 积分结束时的状态
		- phi1: 积分结束时的切向量矩阵
	"""
	# 将状态和切向量合并为扩展状态向量
	y0 = np.hstack([s0, phi0.ravel()])
	
	# 积分扩展系统
	res = solve_ivp(
		fun=lambda t, y: _extended_rhs(t, y, p),
		y0=y0,
		t_span=(t0, t1),
		method=p.method,
		rtol=p.rtol,
		atol=p.atol,
		dense_output=False,  # 不需要密集输出，只需最终值
	)
	if not res.success:
		raise RuntimeError("MLE 分段积分失败: " + str(res.message))
	
	# 提取最终状态和切向量
	final = res.y[:, -1]
	s1 = final[:3]              # 最终状态
	phi1 = final[3:].reshape(3, 3)  # 最终切向量矩阵
	return s1, phi1


def compute_mle(
	p: ChuaParams,
	x0: np.ndarray | None = None,
	t_burn: float | None = None,
	t_total: float | None = None,
	qr_dt: float | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
	"""使用 QR 重正交方法计算最大 Lyapunov 指数。
	
	实现基于变分方程和 QR 重正交的 Benettin/Wolf 方法：
	1. 预热阶段：仅积分基流至 t_burn
	2. 分段积分扩展系统（状态+切向量）
	3. 每段后进行 QR 重正交
	4. 累积对数并计算运行估计
	
	参数:
		p: 参数对象
		x0: 初始状态（None 时随机生成）
		t_burn: 预热时间（默认 p.t_burn）
		t_total: 总积分时间（默认 p.mle_total）
		qr_dt: QR 重正交间隔（默认 p.qr_dt）
		
	返回:
		(lambda1, times, running): 元组
		- lambda1: 最终估计的最大 Lyapunov 指数
		- times: 时间序列数组
		- running: 运行估计序列数组（随时间的收敛过程）
	"""
	# 初始化参数
	if x0 is None:
		rng = np.random.default_rng(p.seed)
		x0 = rng.normal(scale=0.1, size=3)

	if t_burn is None:
		t_burn = p.t_burn
	if t_total is None:
		t_total = p.mle_total
	if qr_dt is None:
		qr_dt = p.qr_dt

	# 步骤 1: 预热阶段（仅积分基流，不积分变分方程）
	from .integrate import integrate_trajectory
	res_burn = integrate_trajectory(p, x0, (0.0, t_burn), with_events=False, dense_output=False)
	if not res_burn.success:
		raise RuntimeError("预热失败: " + str(res_burn.message))
	s = res_burn.y[:, -1]  # 获取预热末状态

	# 步骤 2-3: 扩展系统积分与 QR 重正交
	phi = np.eye(3)  # 初始切向量矩阵为单位阵
	log_accum = 0.0  # 累积对数
	est_times = []   # 时间序列
	running = []     # 运行估计序列
	
	t = 0.0
	segments = int(np.ceil(t_total / qr_dt))  # 计算分段数
	
	for k in range(segments):
		# 当前积分段的时间区间
		t0 = t_burn + t
		t1 = t_burn + min(t + qr_dt, t_total)
		
		# 在 [t0, t1] 上积分扩展系统（状态+切向量）
		s, phi = _propagate_segment(p, s, phi, t0, t1)
		
		# QR 重正交：phi = Q * R，用 Q 作为下一段的初始值
		Q, R = np.linalg.qr(phi)
		phi = Q
		
		# 累积 R 的第一对角元的对数
		log_accum += np.log(np.abs(np.diag(R)[0]))
		
		# 更新时间和计算运行估计
		t = min(t + qr_dt, t_total)
		lambda1 = log_accum / t  # 运行估计：λ₁(t) = S_N / T_N
		
		est_times.append(t)
		running.append(lambda1)

	return float(lambda1), np.asarray(est_times), np.asarray(running)


