# 基于深度学习的量子轨线数据概率分布函数高精度评估
项目简介
随着量子计算与量子信息科学的飞速发展，处理高维、复杂的量子系统数据成为科研痛点。传统的概率密度函数（PDF）评估方法（如核密度估计 KDE）在面对非高斯、多峰分布的量子轨线数据时，往往面临带宽选择困难、边界效应显著及高维计算成本高昂的问题。
本项目旨在利用深度学习中的前沿技术——**归一化流**，构建一种高精度的 PDF 估计算法。通过对比传统统计学方法，验证其在量子轨线数据处理上的优越性，为精确计算量子演化提供基础分布函数支持。
核心创新点
1.  **物理驱动的数据生成**：使用 `QuTiP` 的 `mcsolve` 生成具有量子跳跃和随机性的蒙特卡洛轨迹，而非确定性演化。
2.  **算法锁定**：采用 `RealNVP` (Real-valued Non-Volume Preserving) 归一化流，显式建模概率密度，直接计算似然函数，物理意义明确。
3.  **工程化架构**：精心设计，严格分离数据、模型、训练与评估，确保可复现性。
---
## 🛠️ 技术栈
| 类别 | 核心工具/库 | 用途说明 |
| :--- | :--- | :--- |
| **物理模拟** | [QuTiP](http://qutip.org/) | 量子系统演化模拟（求解薛定谔方程与蒙特卡洛轨迹） |
| **深度学习** | [PyTorch](https://pytorch.org/) | 深度学习框架，自动微分 |
| **核心算法** | [nflows](https://github.com/bayesiains/nflows) | 提供归一化流模型（如 RealNVP, MAF） |
| **基准算法** | SciPy | 实现传统核密度估计 (KDE) 作为对比基线 |
| **可视化** | Matplotlib, Seaborn | 出版级绘图，Science 风格配色 |
| **文档排版** | LaTeX (Thuthesis) | 符合规范的毕业论文撰写 |
---

```text
DL_Quantum_Trajectory_PDF_Estimation/
├── README.md                          # 项目说明书
├── requirements.txt                   # 依赖清单：pip install -r requirements.txt
├── environment.yml                    # Conda环境配置：保证可复现性
├── configs/                           # 【配置层】超参数管理
│   ├── base_config.yaml               # 随机种子、设备(CUDA/CPU)、数据路径
│   ├── model_config.yaml              # Flow类型、隐藏层维度、层数
│   └── train_config.yaml              # Batch Size、Learning Rate、Epochs
├── data/                              # 【数据层】静态数据
│   ├── raw/                           # 原始数据 .npy/.h5 (由QuTiP生成)
│   ├── processed/                     # 训练/验证/测试集
│   └── external/                      # 外部公开数据
├── src/                               # 【源代码层】核心算法
│   ├── data/                          # 模块1：数据
│   │   ├── quantum_simulator.py       # <关键> QuTiP mcsolve 封装
│   │   ├── dataset_loader.py          # PyTorch DataLoader
│   │   └── preprocessing.py           # 归一化、清洗
│   ├── models/                        # 模块2：模型
│   │   ├── normalizing_flow.py        # <关键> RealNVP 模型封装
│   │   ├── baselines.py               # KDE 基准封装
│   │   └── layers.py                  # 自定义层
│   ├── engine/                        # 模块3：引擎
│   │   ├── trainer.py                 # 训练循环
│   │   ├── evaluator.py               # 评估逻辑 (KL, LogP)
│   │   └── utils.py                   # 辅助工具
│   └── visualization/                 # 模块4：绘图
│       └── plotter.py                 # PDF对比图、Loss曲线
├── notebooks/                         # 【探索层】Jupyter 调试
│   ├── 01_data_exploration.ipynb      # 数据检查
│   ├── 02_kde_baseline.ipynb         # KDE 实验
│   └── 03_model_debug.ipynb          # 模型调试
├── scripts/                           # 【执行层】脚本
│   ├── run_generate_data.sh
│   ├── run_train.sh
│   └── run_eval.sh
├── outputs/                           # 【输出层】结果 (不入Git)
│   ├── figures/                       # 论文配图
│   ├── checkpoints/                   # 模型权重
│   └── logs/                          # 训练日志
├── thesis/                            # 【论文层】LaTeX 源码
│   ├── main.tex                       # 主文件
│   ├── chapters/                      # 章节
│   ├── figures/                       # 图片
│   └── references.bib                 # 参考文献
└── docs/                              # 【文档层】过程管理
    ├── literature_review.md
    ├── weekly_reports/                # 每周进度
    └── meeting_notes.md
    
快速开始
1. 环境配置
建议使用 Anaconda 创建虚拟环境：
# 创建环境
conda create -n quantum_ai python=3.9
conda activate quantum_ai
# 安装核心依赖
pip install -r requirements.txt
2. 生成量子轨线数据
使用 QuTiP 模拟双势阱系统或非线性谐振子，生成蒙特卡洛轨迹数据。
# 根据实际脚本扩展名运行或
python scripts/run_generate_data.py
说明：代码将调用 qutip.mcsolve 生成随机量子轨迹，并计算 $P(x) = |\psi(x)|^2$ 作为 Ground Truth。
3. 训练归一化流模型
使用生成的数据训练 RealNVP 模型：
python scripts/run_train.py --config configs/train_config.yaml
核心逻辑：损失函数为负对数似然 loss = -model.log_prob(batch).mean()。
4. 评估与可视化
对比真实分布、KDE 基准与深度学习模型的拟合效果：
python scripts/run_eval.py
输出图表将保存至 outputs/figures/，包含 KL 散度、Wasserstein 距离等关键指标。
📚 理论基础
1. 物理模拟层
本项目的数据源是量子蒙特卡洛轨迹。与确定性演化（mesolve）不同，mcsolve 能够捕捉系统演化中的随机性和量子跳跃。
哈密顿量示例（谐振子）：
$$ H = \frac{p^2}{2m} + \frac{1}{2}m\omega^2 x^2 $$
2. 核心算法层：归一化流
归一化流通过一系列可逆变换 $f$ 将简单分布（如高斯分布 $z$）映射到复杂数据分布 $x$。
变量变换公式：
$$ p_x(x) = p_z(z) \left| \det \left( \frac{\partial f^{-1}(x)}{\partial x} \right) \right| $$
RealNVP 核心机制：
使用仿射耦合层，将输入分为两部分 $x_1, x_2$：
$$
\begin{aligned}
y_1 &= x_1 \
y_2 &= x_2 \cdot \exp(s(x_1)) + t(x_1)
\end{aligned}
$$
这种设计保证了雅可比行列式是三角矩阵，极易计算，使得极大似然估计（MLE）训练成为可能。
执行计划 (75天冲刺)
参考文献
Dinh et al. "Density estimation using Real NVP", *ICLR 2017*.
Papamakarios et al. "Normalizing Flows for Probabilistic Modeling and Inference", *JMLR 2021*.
QuTiP Documentation: qutip.org
Thuthesis: 大学学位论文模板
📄 License
本项目采用 MIT 许可证。详见 LICENSE 文件。
👨‍💻 作者与致谢
本项目由 [你的名字] 开发，作为本科/研究生毕业设计。感谢 Quantum Optics 社区及 PyTorch 团队的开源贡献。
<div align="center">
如果这个项目对你有帮助，请给一个 ⭐️ Star 支持一下！
</div>
