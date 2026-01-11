# Deep-Learning-Quantum-Trajectory-Data-Probability-Distribution-Function-Estimation
基于深度学习的量子轨线数据概率分布函数高精度评估
text
DL_Quantum_Trajectory_PDF_Estimation/
    ├── README.md # 项目说明书
    ├── requirements.txt # 依赖清单
    ├── environment.yml # Conda环境配置 (保证可复现性)
    ├── configs/ # 【配置层】超参数管理
    │    ├── base_config.yaml # 随机种子、设备(CUDA/CPU)、数据路径
    │    ├── model_config.yaml # Flow类型、隐藏层维度、层数
    │    └── train_config.yaml # Batch Size、Learning Rate、Epochs
    ├── data/ # 【数据层】静态数据
    │├── raw/ # 原始数据 .npy/.h5 (由QuTiP生成)
    │    ├── processed/ # 训练/验证/测试集
    │    └── external/ # 外部公开数据
    ├── src/ # 【源代码层】核心算法
    │├── data/ # 模块1：数据
    │    │    ├── quantum_simulator.py # <关键> QuTiP mcsolve 封装
    │    │    ├── dataset_loader.py # PyTorch DataLoader
    │    │    └── preprocessing.py # 归一化、清洗
    │    ├── models/ # 模块2：模型
    │    │    ├── normalizing_flow.py # <关键> RealNVP 模型封装
    │    │    ├── baselines.py # KDE 基准封装
    │    │    └── layers.py # 自定义层
    │    ├── engine/ # 模块3：引擎
    │    │    ├── trainer.py # 训练循环
    │    │    ├── evaluator.py # 评估逻辑 (KL散度, LogP)
    │    │    └── utils.py # 辅助工具
    │    └── visualization/ # 模块4：绘图
    │    └── plotter.py # PDF对比图、Loss曲线
    ├── notebooks/ # 【探索层】Jupyter 调试
    │    ├── 01_data_exploration.ipynb # 数据检查
    │    ├── 02_kde_baseline.ipynb # KDE 实验
    │    └── 03_model_debug.ipynb # 模型调试
    ├── scripts/ # 【执行层】脚本
    │    ├── run_generate_data.sh
    │    ├── run_train.sh
    │    └── run_eval.sh
    ├── outputs/ # 【输出层】结果 (不入Git)
    │    ├── figures/ # 论文配图
    │    ├── checkpoints/ # 模型权重
    │    └── logs/ # 训练日志
    ├── thesis/ # 【论文层】LaTeX 源码
    │    ├── main.tex # 主文件
    │    ├── chapters/ # 章节
    │    ├── figures/ # 图片
    │    └── references.bib # 参考文献
    └── docs/ # 【文档层】过程管理
    ├── literature_review.md
    ├── weekly_reports/ # 每周进度
    └── meeting_notes.md

建议使用 Anaconda 创建虚拟环境：
创建环境：conda create -n quantum_ai python=3.10.11
        conda activate quantum_ai
安装核心依赖：pip install -r requirements.txt



