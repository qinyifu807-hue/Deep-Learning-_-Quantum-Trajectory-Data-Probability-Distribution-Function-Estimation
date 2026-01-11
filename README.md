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
├── requirements.txt                   # 依赖清单requirements.txt
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
    

