# 分段线性蔡氏电路的数值仿真与动力学分析

本项目实现了分段线性（PWL）蔡氏电路的完整数值仿真与动力学分析框架，包括混沌吸引子仿真、最大 Lyapunov 指数计算、分岔分析、Poincaré 截面分析以及多元数值验证方法。

## 项目简介

蔡氏电路是能够产生混沌行为的最简单自治电路之一。本项目针对分段线性非线性在切换点处不可微的问题，构建了一套兼顾精度、鲁棒性与可复现性的数值分析与验证框架。

### 主要特性

- **平滑 Jacobian 策略**：采用 $|x|\mapsto\sqrt{x^2+\epsilon^2}$ 的平滑近似处理分段线性非线性，提高变分方程数值稳定性
- **多元数值验证**：集成积分器交叉验证（RK45 vs Radau）、KCL 残差分析、自相关函数（ACF）和功率谱密度（PSD）分析
- **系统动力学分析**：包括双涡卷吸引子、最大 Lyapunov 指数、分岔图、Poincaré 截面、Shannon 熵等完整分析
- **可复现性**：所有随机数生成基于固定种子，参数配置与结果以 JSON 格式保存

## 安装要求

### 系统要求

- Python 3.8 或更高版本
- Windows、Linux 或 macOS

### 依赖库

安装所需依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `numpy >= 1.26`：数值计算
- `scipy >= 1.11`：数值积分（solve_ivp）
- `matplotlib >= 3.8`：可视化
- `pandas >= 2.1`：数据处理

可选依赖：
- `schemdraw >= 0.15`：用于生成电路原理图（可选）

## 快速开始

### 运行主实验

执行以下命令即可复现论文中的所有核心结果：

```bash
python main.py
```

该脚本将自动创建 `output/` 目录并生成所有图表与数据文件，包括：

- **轨迹与相图**：`time_phase.png`、`fig3_chua_attractor.png`
- **Lyapunov 指数**：`mle_convergence.png`、`mle.json`
- **分岔图**：`bifurcation.png`、`bifurcation.csv`
- **Poincaré 截面**：`poincare_scatter.png`、`poincare_section.csv`
- **统计分析**：`fig_stats.pdf`（PSD 和 ACF）
- **数值验证**：`fig_validation.pdf`、`integrator_crosscheck.json`
- **残差分析**：`residual.png`、`residual.csv`

典型运行时间约 5-10 分钟（取决于硬件配置），主要耗时在 MLE 计算与分岔图扫描。

### 运行其他实验

#### 平滑参数敏感性分析

```bash
python epsilon_sensitivity.py
```

生成平滑参数 $\epsilon$ 对最大 Lyapunov 指数的影响曲线（对应论文图 6）。

#### 参数平面扫描

```bash
python param_plane_mle.py
```

在 $(\alpha, m_1)$ 参数平面上扫描最大 Lyapunov 指数分布（对应论文图 8）。

## 项目结构

```
分段线性蔡氏电路的数值仿真与动力学分析/
├── chua_py/              # 核心库
│   ├── __init__.py       # 包初始化
│   ├── config.py         # 参数配置（ChuaParams）
│   ├── model.py          # 数学模型（状态方程、雅可比矩阵）
│   ├── integrate.py      # 数值积分
│   ├── lyapunov.py       # Lyapunov 指数计算
│   ├── bifurcation.py    # 分岔分析
│   ├── stats.py          # 统计分析
│   ├── validation.py     # 数值验证
│   ├── robustness.py     # 鲁棒性分析
│   └── plotting.py       # 可视化函数
├── main.py               # 主实验脚本
├── epsilon_sensitivity.py # 平滑参数敏感性分析
├── param_plane_mle.py    # 参数平面扫描
├── requirements.txt      # 依赖列表
├── .gitignore           # Git 忽略文件
└── README.md            # 本文件
```

## 核心模块说明

### chua_py 库

核心库 `chua_py/` 提供以下功能模块：

- **config.py**：定义 `ChuaParams` 数据类，集中管理所有系统参数与数值控制参数
- **model.py**：实现蔡氏电路的数学模型，包括分段线性非线性函数、状态方程右端项、雅可比矩阵（含平滑版本）以及 Poincaré 截面事件检测
- **integrate.py**：基于 SciPy 的 `solve_ivp` 封装轨迹积分功能，提供均匀采样和 KCL 残差计算
- **lyapunov.py**：实现最大 Lyapunov 指数计算，采用基于变分方程和 QR 重正交的 Benettin/Wolf 方法
- **bifurcation.py**：提供参数扫描与分岔图生成功能
- **stats.py**：实现统计分析工具，包括自相关函数、积分自相关时间和 Shannon 熵计算
- **validation.py**：提供数值验证功能，对比不同积分器的结果一致性
- **robustness.py**：实现蒙特卡洛鲁棒性分析
- **plotting.py**：包含所有可视化函数，生成论文中使用的各类图表

## 使用示例

### 自定义参数

用户可通过修改 `ChuaParams` 实例自定义参数：

```python
from chua_py import ChuaParams, compute_mle

# 修改电路参数
p = ChuaParams(alpha=12.0, m1=-0.65)
lambda1, times, running = compute_mle(p)
print(f"最大 Lyapunov 指数: {lambda1:.5f}")
```

### 模块化使用

核心库设计为可独立使用的模块，用户可根据需要调用特定功能：

```python
from chua_py import ChuaParams, integrate_trajectory, sample_dense
import numpy as np

# 计算单条轨迹
p = ChuaParams()
x0 = np.array([0.1, 0.0, 0.0])
res = integrate_trajectory(p, x0, (0.0, 100.0))
t, states = sample_dense(res, 50.0, 100.0, 0.02)
```

## 数值结果

在典型参数下（$\alpha=10.0$, $\beta=14.87$, $\rho=0.0$, $m_0=-1.143$, $m_1=-0.714$），本框架得到的主要数值结果：

- **最大 Lyapunov 指数**：$\lambda_1 \approx 0.00580 > 0$（表明系统为混沌）
- **Poincaré 截面 Shannon 熵**：$H \approx 2.60$ nats
- **积分自相关时间**：$\tau_{\text{int}} \approx 134.39$
- **吸引子结构**：对称的双涡卷混沌吸引子

详细结果与图表见 `output/` 目录。

## 代码可复现性

- 所有随机数生成均基于固定种子（默认 `seed=42`），确保在相同参数配置下结果完全可复现
- 参数配置、数值结果与元数据均以 JSON 格式保存，便于后续分析与对比
- 代码采用纯 Python 实现，不依赖专有软件或特殊硬件

## 相关论文

本项目代码对应论文：

**分段线性蔡氏电路的数值仿真与动力学分析**

如使用本代码，请引用相关论文。

## 许可证

本项目采用 MIT 许可证（详见 LICENSE 文件）。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---



