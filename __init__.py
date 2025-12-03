"""蔡氏电路仿真包（纯 Python 实现，基于 NumPy/SciPy/Matplotlib）。

本包提供了蔡氏电路数值仿真的完整功能，包括：
- 数学模型：状态方程、雅可比矩阵、分段线性非线性函数
- 数值积分：轨迹积分、密集采样、物理残差计算
- Lyapunov 指数：基于变分方程和 QR 重正交的最大 Lyapunov 指数计算
- 分岔分析：参数扫描与 Poincaré 截面数据收集
- 统计分析：自相关函数、积分自相关时间、Shannon 熵
- 数值验证：积分器交叉验证、蒙特卡洛鲁棒性分析
- 可视化：各类图表生成函数

本模块导出所有主要函数和类，方便外部脚本直接导入使用。
"""

from .config import ChuaParams
from .model import f_piecewise, rhs, jacobian, poincare_event
from .integrate import integrate_trajectory, sample_dense, kcl_residual
from .lyapunov import compute_mle
from .bifurcation import scan_parameter
from .validation import cross_check_integrators
from .stats import compute_acf, integrated_autocorrelation_time, shannon_entropy
from .robustness import monte_carlo_mle
from .plotting import (
    plot_circuit_schematic,
    plot_state_space_block,
    plot_attractor_3d,
    plot_stats_panels,
    plot_validation_panels,
    compute_psd_series,
)

__all__ = [
	"ChuaParams",
	"f_piecewise",
	"rhs",
	"jacobian",
	"poincare_event",
	"integrate_trajectory",
	"sample_dense",
	"kcl_residual",
	"compute_mle",
	"scan_parameter",
	"cross_check_integrators",
	"compute_acf",
	"integrated_autocorrelation_time",
	"shannon_entropy",
	"monte_carlo_mle",
	"plot_circuit_schematic",
	"plot_state_space_block",
	"plot_attractor_3d",
	"plot_stats_panels",
	"plot_validation_panels",
	"compute_psd_series",
]

__version__ = "0.1.0"


