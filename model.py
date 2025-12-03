from __future__ import annotations

import numpy as np
from typing import Tuple

from .config import ChuaParams


def f_piecewise(x: np.ndarray | float, p: ChuaParams) -> np.ndarray | float:
	"""分段线性非线性函数，对应论文中式(5)。
	
	使用精确的绝对值函数 |x| 实现分段线性特性，用于主积分过程。
	该函数在 x = ±1 处有不可微的切换点。
	
	参数:
		x: 输入变量，可以是标量或数组
		p: 参数对象，需要包含 m0 和 m1 属性
		
	返回:
		f(x) = m1*x + 0.5*(m0-m1)*(|x+1| - |x-1|)
		当 |x| < 1 时，f'(x) = m0
		当 |x| >= 1 时，f'(x) = m1
	"""
	m0, m1 = p.m0, p.m1
	return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))


def _soft_abs(x: np.ndarray | float, eps: float) -> np.ndarray | float:
	"""平滑绝对值函数：|x| ≈ sqrt(x² + ε²)
	
	这是一个内部辅助函数，用于构造平滑的非线性函数。
	当 ε 很小时，该函数在远离原点处与 |x| 几乎相同，在原点附近平滑过渡。
	
	参数:
		x: 输入变量，可以是标量或数组
		eps: 平滑参数，控制平滑程度
		
	返回:
		sqrt(x² + ε²)，平滑版本的 |x|
	"""
	return np.sqrt(x * x + eps * eps)


def f_smooth_and_derivative(x: float, p: ChuaParams) -> Tuple[float, float]:
	"""平滑版本的 f(x) 及其导数 f'(x)，用于 MLE 计算。
	
	通过将 |x| 替换为 sqrt(x² + ε²) 来实现平滑，使得导数在切换点 x = ±1 附近连续。
	该函数仅在变分方程的雅可比矩阵计算中使用，以提高数值稳定性。
	
	参数:
		x: 输入变量（标量）
		p: 参数对象，需要包含 eps_smooth, m0, m1 属性
		
	返回:
		(f, fp): 元组，包含平滑函数值 f 和其导数 fp
	"""
	eps = p.eps_smooth
	m0, m1 = p.m0, p.m1
	# 计算平滑绝对值：|x+1| 和 |x-1| 的平滑版本
	sa1 = _soft_abs(x + 1.0, eps)
	sa2 = _soft_abs(x - 1.0, eps)
	# 平滑函数值
	f = m1 * x + 0.5 * (m0 - m1) * (sa1 - sa2)
	# 平滑导数：通过链式法则计算
	fp = m1 + 0.5 * (m0 - m1) * ((x + 1.0) / sa1 - (x - 1.0) / sa2)
	return f, fp


def rhs(t: float, s: np.ndarray, p: ChuaParams) -> np.ndarray:
	"""状态方程右端项，对应论文中式(4)。
	
	计算蔡氏电路状态方程的三个分量：
		ẋ = α(y - x - f(x))
		ẏ = x - y + z
		ż = -βy - ρz
	
	参数:
		t: 时间（无量纲，本函数中未使用，但保持接口一致性）
		s: 状态向量 [x, y, z]，形状为 (3,)
		p: 模型参数对象
		
	返回:
		状态导数 [ẋ, ẏ, ż]，形状为 (3,) 的 numpy 数组
	"""
	x, y, z = s
	# 计算分段线性非线性函数 f(x)
	f = f_piecewise(x, p)
	# 状态方程的三个分量
	dx = p.alpha * (y - x - f)  # ẋ = α(y - x - f(x))
	dy = x - y + z              # ẏ = x - y + z
	dz = -p.beta * y - p.rho * z  # ż = -βy - ρz
	return np.array([dx, dy, dz], dtype=float)


def jacobian(t: float, s: np.ndarray, p: ChuaParams, smooth: bool = True) -> np.ndarray:
	"""雅可比矩阵 J(x)，对应论文中式(8)。
	
	计算状态方程关于状态的雅可比矩阵，用于变分方程和 Lyapunov 指数计算。
	
	当 smooth=True（默认）时，使用平滑的 f'(x) 在 |x|=1 附近，以提高变分方程
	的数值稳定性。当 smooth=False 时，使用严格的分段导数。
	
	参数:
		t: 时间（未使用，保持接口一致性）
		s: 状态向量 [x, y, z]
		p: 参数对象
		smooth: 是否使用平滑导数（默认 True）
		
	返回:
		3×3 雅可比矩阵，numpy 数组
	"""
	x, y, z = s
	if smooth:
		# 使用平滑导数（用于变分方程，提高数值稳定性）
		_, fp = f_smooth_and_derivative(x, p)
	else:
		# 严格分段导数：|x| < 1 时为 m0，否则为 m1
		fp = p.m0 if np.abs(x) < 1.0 else p.m1

	# 构造雅可比矩阵，对应式(8)
	J11 = -p.alpha * (1.0 + fp)  # ∂(ẋ)/∂x = -α(1 + f'(x))
	J12 = p.alpha                 # ∂(ẋ)/∂y = α
	J13 = 0.0                     # ∂(ẋ)/∂z = 0
	J21 = 1.0                     # ∂(ẏ)/∂x = 1
	J22 = -1.0                    # ∂(ẏ)/∂y = -1
	J23 = 1.0                     # ∂(ẏ)/∂z = 1
	J31 = 0.0                     # ∂(ż)/∂x = 0
	J32 = -p.beta                 # ∂(ż)/∂y = -β
	J33 = -p.rho                  # ∂(ż)/∂z = -ρ
	return np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]], dtype=float)


def poincare_event(t: float, s: np.ndarray, p: ChuaParams) -> float:
	"""Poincaré 截面事件检测函数。
	
	定义 Poincaré 截面为 y = 0，且要求 y 从负到正穿越（向上穿越）。
	该函数返回 y 的值，当 y = 0 时事件触发。
	
	参数:
		t: 时间
		s: 状态向量 [x, y, z]
		p: 参数对象（未使用，但保持接口一致性）
		
	返回:
		y 的值，当返回值为 0 时事件触发
	"""
	# 根在 y=0；direction=+1 确保仅检测向上穿越（y 从负到正）
	return s[1]


# 为 SciPy solve_ivp 附加事件配置属性
poincare_event.direction = 1.0  # type: ignore[attr-defined]
"""事件方向：+1 表示仅检测从负到正的穿越（向上穿越）"""

poincare_event.terminal = False  # type: ignore[attr-defined]
"""事件是否终止积分：False 表示非终止事件，积分继续进行"""


